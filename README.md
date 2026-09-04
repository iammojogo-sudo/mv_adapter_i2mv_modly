# MV-Adapter (Multiview)

A Modly extension that generates 6 orthographic reference views (front, right, back, left, top, bottom) from a single input image using [MV-Adapter I2MV-SD2.1](https://github.com/huanngzh/MV-Adapter) at 512×512 resolution.

The SD2.1 variant is the low-VRAM build: it is the one the MV-Adapter authors recommend for GPUs with **less than 6 GB** of VRAM (see their README). It uses CPU offloading of the text/VAE components so the UNet + activations fit on a 6 GB card.

> Note: MV-Adapter has no SD 1.5 weight. The smallest available base is SD 2.1 (`mvadapter_i2mv_sd21.safetensors`), so we switched from the SDXL variant (which needed ~8 GB just for its UNet and OOM'd on this GPU).

## Pipeline

1. **Install** — `setup.py` creates an isolated venv and installs dependencies
2. **Download weights** — Click "install weights" on the **Generate Reference Views** node; it fetches both the MV-Adapter adapter (~0.7 GB) and the SD2.1 base model (~2 GB, ungated mirror)
3. **Generate** — The first run generates 6 views + a 1×6 grid image, ready to feed into a 3D mesh generator

## Files

| File | Purpose |
|------|---------|
| `setup.py` | Venv creation and dependency install |
| `generator.py` | Modly generator — runs bridge in subprocess |
| `bridge.py` | MV-Adapter i2mv SD2.1 pipeline (Plucker camera embeddings, CPU offload) |
| `manifest.json` | Modly extension manifest |

## Credits

- [MV-Adapter](https://github.com/huanngzh/MV-Adapter) by Zehuan Huang et al. (ICCV 2025)
- [stable-diffusion-2-1-base](https://huggingface.co/Manojb/stable-diffusion-2-1-base) (ungated mirror) — SD2.1 base model
