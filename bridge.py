"""MV-Adapter bridge for multi-view image generation.

i2mv mode: MV-Adapter-I2MV-SD2.1 at 512x512, Plucker camera embeddings.
The SD2.1 variant is the low-VRAM build recommended for <6GB GPUs.
Generates 6 orthographic views on a horizontal sweep (all elevation 0, the
i2mv model's trained configuration) and saves them as named direction files
(front, front-left, left, back, right, front-right) plus a 2×2 grid of the 4
cardinal views in reading order [front, left, back, right] for the downstream
mesh generator. i2mv cannot produce real top/bottom views.
"""

import json
import math
import os
import sys
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import (
    MVAdapterI2MVSDPipeline,
)
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras_ortho
from diffusers import DDPMScheduler


# ── Camera math ─────────────────────────────────────────────────────────

def get_c2w(elevation_deg, distance, azimuth_deg, device=None):
    if isinstance(elevation_deg, list):
        elevation_deg = torch.tensor(elevation_deg, dtype=torch.float32, device=device)
    if isinstance(azimuth_deg, list):
        azimuth_deg = torch.tensor(azimuth_deg, dtype=torch.float32, device=device)
    if isinstance(distance, list):
        distance = torch.tensor(distance, dtype=torch.float32, device=device)
    num_views = len(elevation_deg)
    elev_r = elevation_deg * math.pi / 180
    azim_r = azimuth_deg * math.pi / 180
    cam_pos = torch.stack([
        distance * torch.cos(elev_r) * torch.cos(azim_r),
        distance * torch.cos(elev_r) * torch.sin(azim_r),
        distance * torch.sin(elev_r),
    ], dim=-1)
    center = torch.zeros_like(cam_pos)
    up = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)[None].repeat(num_views, 1)
    lookat = F.normalize(center - cam_pos, dim=-1)
    right = F.normalize(torch.cross(lookat, up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)
    c2w3x4 = torch.cat([torch.stack([right, up, -lookat], dim=-1), cam_pos[:, :, None]], dim=-1)
    c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def make_image_grid(images, rows=None, cols=None, resize=None):
    if rows is None and cols is not None:
        assert len(images) % cols == 0
        rows = len(images) // cols
    elif cols is None and rows is not None:
        assert len(images) % rows == 0
        cols = len(images) // rows
    elif rows is None and cols is None:
        cols = int(math.sqrt(len(images)))
        rows = (len(images) + cols - 1) // cols
    assert len(images) == rows * cols
    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def tensor_to_pil(data, batched=False):
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if data.dtype in (np.float32, np.float16):
        data = (data * 255).astype(np.uint8)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    if batched:
        return [Image.fromarray(d) for d in data]
    return Image.fromarray(data)


# ── MV-Adapter camera config ───────────────────────────────────────────

I2MV_RESOLUTION = 512
MV_DISTANCE = 1.8


def run_mv_adapter(mesh_path, ref_image_path, output_dir, device="cuda",
                   text="high quality", remove_bg=True, num_steps=50,
                   guidance_scale=3.0, seed=-1, model_dir=None, resolution=None):
    os.makedirs(output_dir, exist_ok=True)
    resolution = int(resolution or I2MV_RESOLUTION)
    # Horizontal sweep (all elevation 0). Direction names follow the MV-Adapter
    # paper's canonical order for the trained azimuths {0,45,90,180,270,315}:
    #   front, front-left, left, back, right, front-right.
    # The official i2mv script shifts those azimuths by -90, which yields the
    # same direction sequence in output order (verified: view_0 = front, view_3
    # = back). The 2×2 grid below uses only the 4 cardinal views in reading order
    # [front, left, back, right], matching the downstream mesh generator.
    names = ["front", "front-left", "left", "back", "right", "front-right"]
    GRID_2X2 = [0, 2, 3, 4]  # front, left, back, right

    local_base = None
    local_adapter = None
    if model_dir:
        import glob as _glob
        _cand_base = (
            Path(model_dir),
            Path(model_dir).parent / "sd21-base",
        )
        for c in _cand_base:
            if (c / "model_index.json").exists():
                local_base = str(c)
                break
        _cand_adapter = _glob.glob(
            str(Path(model_dir) / "**" / "mvadapter_i2mv_sd21.safetensors"), recursive=True)
        if _cand_adapter:
            local_adapter = _cand_adapter[0]

    if not local_base:
        raise RuntimeError(
            "SD 2.1 base model not found. Please install weights via the Modly extension panel "
            "(mv-adapter node) before generating."
        )

    # Generate Plucker camera embeddings
    print(json.dumps({"type": "log", "message": "Generating camera Plucker embeddings"}), flush=True)
    # i2mv is a HORIZONTAL-SWEEP model: it was trained with all cameras at
    # elevation 0 and cannot produce real top/bottom views (feeding it ±90° just
    # yields garbage side views). Use the exact camera config from the official
    # inference_i2mv_sd.py script so all 6 views are valid and the model performs
    # as intended. These views feed downstream mesh generation.
    i2mv_elevs = [0, 0, 0, 0, 0, 0]
    i2mv_azims = [x - 90 for x in [0, 45, 90, 180, 270, 315]]
    print(json.dumps({"type": "log",
        "message": "i2mv camera sweep (elev=0): " +
        ", ".join(f"{n}={a}deg" for n, a in zip(names, i2mv_azims))}), flush=True)
    c2w = get_c2w(i2mv_elevs, [MV_DISTANCE] * 6, i2mv_azims, device=device)
    plucker_embeds = get_plucker_embeds_from_cameras_ortho(c2w, [1.1] * 6, resolution)
    control_images = ((plucker_embeds + 1.0) / 2.0).clamp(0, 1).to(device=device)

    # Load pipeline
    print(json.dumps({"type": "log", "message": f"Loading SD 2.1 base from: {local_base}"}), flush=True)
    pipe = MVAdapterI2MVSDPipeline.from_pretrained(
        local_base, torch_dtype=torch.float16, variant="fp16",
    )
    # The authors find DDPMScheduler wrapped with a shift-SNR schedule works
    # best for the SD2.1-based adapters.
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(num_views=6)

    if local_adapter:
        print(json.dumps({"type": "log", "message": f"Loading adapter from local: {local_adapter}"}), flush=True)
        pipe.load_custom_adapter(
            os.path.dirname(local_adapter),
            weight_name=os.path.basename(local_adapter),
        )
    else:
        print(json.dumps({"type": "log", "message": "Loading adapter from HuggingFace"}), flush=True)
        pipe.load_custom_adapter(
            "huanngzh/mv-adapter",
            weight_name="mvadapter_i2mv_sd21.safetensors",
        )
    # Note: the MV/ref attention weights and cond_encoder are created in fp32 by
    # init_custom_adapter, so we cast the whole pipeline to fp16 on the device.
    # We deliberately do NOT use enable_model_cpu_offload() — it is incompatible
    # with this pipeline's reference-hidden-states caching (accelerate hooks break
    # cache_hidden_states across the ref/denoise passes) and it would not cast the
    # adapter weights. The SD2.1 variant is small enough to run fully on-device
    # (officially recommended for <6GB GPUs). Also do NOT call
    # enable_attention_slicing(): that would replace the custom
    # DecoupledMVRowSelfAttnProcessor processors.
    pipe.to(device=device, dtype=torch.float16)
    pipe.cond_encoder.to(device=device, dtype=torch.float16)

    # Load reference image
    print(json.dumps({"type": "log", "message": "Preparing reference image"}), flush=True)
    ref_img = Image.open(ref_image_path).convert("RGB")
    w, h = ref_img.size
    s = min(w, h)
    ref_img = ref_img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    ref_img = ref_img.resize((resolution, resolution), Image.LANCZOS)

    if remove_bg:
        try:
            import onnxruntime  # noqa: F401  (rembg needs this; skip if missing)
            from rembg import remove as rembg_remove
            ref_img = rembg_remove(ref_img, post_process=True)
            ref_img = ref_img.convert("RGBA")
            bg = Image.new("RGBA", ref_img.size, (128, 128, 128, 255))
            ref_img = Image.alpha_composite(bg, ref_img).convert("RGB")
        except BaseException as e:  # rembg may sys.exit(); never abort generation over bg removal
            print(json.dumps({"type": "log", "message": f"rembg unavailable ({e}); generating without background removal"}), flush=True)

    ref_np = np.array(ref_img)
    alpha = np.ones(ref_np.shape[:2], dtype=bool)
    if ref_img.mode == "RGBA":
        alpha = np.array(ref_img)[:, :, 3] > 0
        ref_np = np.array(ref_img.convert("RGB"))
    y, x = np.where(alpha)
    if len(y) > 0:
        y0, y1 = max(y.min() - 2, 0), min(y.max() + 2, ref_np.shape[0])
        x0, x1 = max(x.min() - 2, 0), min(x.max() + 2, ref_np.shape[1])
        crop = ref_np[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        target = int(resolution * 0.9)
        if ch > cw:
            cw = int(cw * target / ch)
            ch = target
        else:
            ch = int(ch * target / cw)
            cw = target
        crop = np.array(Image.fromarray(crop).resize((cw, ch)))
        out = np.full((resolution, resolution, 3), 128, dtype=np.uint8)
        sh = (resolution - ch) // 2
        sw = (resolution - cw) // 2
        out[sh:sh+ch, sw:sw+cw] = crop
        ref_img = Image.fromarray(out)

    ref_img.save(os.path.join(output_dir, "reference_processed.png"))

    # Generate multi-view images
    print(json.dumps({"type": "log", "message": "Running MV-Adapter inference (6 views)"}), flush=True)
    gen_kwargs = {}
    if seed >= 0:
        gen_kwargs["generator"] = torch.Generator(device=device).manual_seed(seed)

    def _step_cb(pipe, step, timestep, callback_kwargs):
        _done = step + 1
        _pct = int(20 + 70 * _done / max(num_steps, 1))
        print(json.dumps({
            "type": "progress",
            "pct": _pct,
            "step": f"Diffusing reference views ({_done}/{num_steps})",
        }), flush=True)
        return callback_kwargs

    images = pipe(
        text,
        height=resolution, width=resolution,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=6,
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=ref_img,
        reference_conditioning_scale=1.0,
        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
        callback_on_step_end=_step_cb,
        callback_on_step_end_tensor_inputs=[],
        **gen_kwargs,
    ).images

    for i, (name, img) in enumerate(zip(names, images)):
        img.save(os.path.join(output_dir, f"{name}.png"))

    # 2×2 grid of the 4 cardinal views in reading order:
    #   [front,  left ]
    #   [back,   right]
    # This matches the downstream mesh generator's expected layout.
    grid_images = [images[i] for i in GRID_2X2]
    grid = make_image_grid(grid_images, rows=2)
    grid.save(os.path.join(output_dir, "grid.png"))

    print(json.dumps({"type": "log", "message": f"Saved 6 views to {output_dir}"}), flush=True)
    return output_dir


if __name__ == "__main__":
    args = json.loads(sys.argv[1])
    mesh_path = args["mesh_path"]
    ref_image_path = args["ref_image_path"]
    output_dir = args.get("output_dir", os.path.join(os.path.dirname(mesh_path), "shape_views"))
    device = args.get("device", "cuda")
    text = args.get("text", "high quality")
    remove_bg = args.get("remove_bg", True)
    num_steps = args.get("num_inference_steps", 50)
    guidance_scale = args.get("guidance_scale", 3.0)
    seed = args.get("seed", -1)
    model_dir = args.get("model_dir", None)
    resolution = args.get("resolution", I2MV_RESOLUTION)

    try:
        run_mv_adapter(mesh_path, ref_image_path, output_dir, device,
                       text, remove_bg, num_steps, guidance_scale, seed, model_dir,
                       resolution=resolution)
        print(json.dumps({"type": "done", "output_dir": output_dir}), flush=True)
    except Exception as e:
        print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
