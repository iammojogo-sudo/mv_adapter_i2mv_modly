import json
import os
import platform
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from services.generators.base import BaseGenerator


EXT_DIR = Path(__file__).resolve().parent


def _venv_python() -> Path:
    is_win = platform.system() == "Windows"
    return EXT_DIR / "venv" / ("Scripts/python.exe" if is_win else "bin/python")


HY_BRIDGE = EXT_DIR / "bridge.py"


class MVAdapterGenerator(BaseGenerator):
    MODEL_ID = "mv-adapter"
    DISPLAY_NAME = "MV-Adapter (Multiview)"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._venv_python = None

    def load(self) -> None:
        self._venv_python = _venv_python()
        if not self._venv_python.exists():
            raise RuntimeError(
                "MV-Adapter venv not found at " + str(self._venv_python) +
                ". Run setup.py first."
            )

    def is_downloaded(self) -> bool:
        check = self.download_check
        has_adapter = (self.model_dir / check).exists() if check else False
        sdxl_base = self.model_dir / "stable-diffusion-xl-base-1.0" / "model_index.json"
        has_base = sdxl_base.exists()
        return has_adapter and has_base

    def _auto_download(self) -> None:
        self._download_weights()

    def _download_weights(self) -> None:
        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 1) MV-Adapter i2mv adapter weight (from the manifest hf_repo)
        if not (self.model_dir / self.download_check).exists():
            print(f"[mv-adapter] Downloading adapter {self.hf_repo} ...")
            snapshot_download(
                repo_id=self.hf_repo,
                local_dir=str(self.model_dir),
                ignore_patterns=["*.md", "LICENSE", "NOTICE", ".gitattributes"],
            )
            print("[mv-adapter] Adapter downloaded.")
        else:
            print("[mv-adapter] Adapter already present.")

        # 2) SDXL base model (required by the i2mv pipeline)
        sdxl_dir = self.model_dir / "stable-diffusion-xl-base-1.0"
        if not (sdxl_dir / "model_index.json").exists():
            print("[mv-adapter] Downloading SDXL base model (~7 GB) ...")
            snapshot_download(
                repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                local_dir=str(sdxl_dir),
                ignore_patterns=["*.md", "LICENSE", "NOTICE", ".gitattributes"],
                local_dir_use_symlinks=False,
            )
            print("[mv-adapter] SDXL base model downloaded.")
        else:
            print("[mv-adapter] SDXL base model already present.")

    def unload(self) -> None:
        self._venv_python = None

    def _resolve_mesh_path(self, rel: Path) -> str:
        """Resolve a mesh path that Modly sends relative to the workspace root.

        The Load 3D Mesh / workflow runner sends mesh_path as
        <workspaceDir>/Workflows/run_…/mesh.glb with the workspace prefix
        stripped, while our outputs_dir is <workspaceDir>/Workflows — so a
        naive join double-counts 'Workflows'. Walk up ancestors and strip any
        overlapping leading segments to locate the real file.
        """
        rel = Path(rel)
        if rel.is_absolute() and rel.exists():
            return str(rel)

        roots: list = []
        for start in (Path(self.outputs_dir).resolve(),
                      Path(self.model_dir).resolve(),
                      Path.cwd().resolve()):
            node = start
            for _ in range(6):
                if node not in roots:
                    roots.append(node)
                if node.parent == node:
                    break
                node = node.parent

        parts = rel.parts
        for root in roots:
            cand = root / rel
            if cand.exists():
                return str(cand)
            # strip overlapping leading segments (e.g. root ends in 'Workflows'
            # and rel starts with 'Workflows')
            for i in range(len(parts)):
                cand = root / Path(*parts[i:])
                if cand.exists():
                    return str(cand)
        # Fallback: best guess
        return str((Path(self.outputs_dir).resolve() / rel).resolve())

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self._venv_python is None:
            self.load()

        run = self._output_dir(params)
        output_dir = run / "views"
        output_dir.mkdir(parents=True, exist_ok=True)
        grid_path = output_dir / "grid.png"

        # Load 3D Mesh node outputs its absolute path in params.filePath.
        # Fall back to mesh_path (used when the mesh comes from another node's
        # output, which Modly sends relative to the workspace).
        # Mesh is OPTIONAL: if not wired, the bridge falls back to plain
        # image-to-multiview (Plucker embeddings) instead of geometry guidance.
        mesh_path_raw = params.get("filePath") or params.get("mesh_path") or ""
        if mesh_path_raw:
            mesh_path = self._resolve_mesh_path(Path(mesh_path_raw))
        elif image_bytes and image_bytes[:4] == b"glTF":
            # Fallback: maybe the mesh was passed as GLB bytes
            mesh_path = str(output_dir / "input_mesh.glb")
            with open(mesh_path, "wb") as f:
                f.write(image_bytes)
        else:
            mesh_path = ""

        ref_path = str(output_dir / "input_ref.png")
        from PIL import Image as _PIL
        import io as _io
        _PIL.open(_io.BytesIO(image_bytes)).save(ref_path)

        num_inference_steps = int(params.get("num_inference_steps", 50))
        guidance_scale = float(params.get("guidance_scale", 3.0))
        remove_bg = params.get("remove_bg", "true") in ("true", "True", True)

        bridge_args = {
            "mesh_path": mesh_path,
            "ref_image_path": ref_path,
            "output_dir": str(output_dir),
            "model_dir": str(self.model_dir),
            "device": "cuda",
            "text": "high quality",
            "remove_bg": remove_bg,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": -1,
        }

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

        if progress_cb:
            if mesh_path:
                progress_cb(10, "Rendering mesh position/normal maps…")
            else:
                progress_cb(10, "Preparing camera embeddings for i2mv…")

        _ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
        captured = []

        def _pump(pipe, cb, sink):
            for raw in pipe:
                line = raw.strip()
                if line:
                    sink.append(line)
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    if line and line[0] in "{[":
                        continue
                    text = _ANSI_RE.sub("", line).replace("\r", "").strip()
                    if text:
                        print(f"[{self.MODEL_ID}] {text}", file=sys.stderr, flush=True)
                    continue
                t = msg.get("type")
                if t == "progress" and cb:
                    cb(msg.get("pct", 0), msg.get("step", ""))
                elif t == "log":
                    text = _ANSI_RE.sub("", msg.get("message", "")).strip()
                    if text:
                        print(f"[{self.MODEL_ID}] {text}", file=sys.stderr, flush=True)

        proc = subprocess.Popen(
            [str(self._venv_python), str(HY_BRIDGE), json.dumps(bridge_args)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        reader = threading.Thread(
            target=_pump, args=(proc.stdout, progress_cb, captured), daemon=True
        )
        reader.start()

        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise RuntimeError("MV-Adapter timed out after 1 hour.")

        reader.join()
        full_out = "\n".join(captured)
        print(f"[{self.MODEL_ID}] Subprocess exit code: {proc.returncode}", file=sys.stderr, flush=True)
        for line in captured[-50:]:
            print(f"[{self.MODEL_ID}] | {line}", file=sys.stderr, flush=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MV-Adapter failed (exit {proc.returncode}): {full_out}")
        if not grid_path.exists():
            raise RuntimeError(f"MV-Adapter output not created: {full_out}")

        if progress_cb:
            progress_cb(100, "Done")
        self.unload()
        return grid_path

    def _output_dir(self, params):
        rf = params.get("run_folder") or params.get("output_dir") or ""
        if rf:
            p = Path(rf)
            if not p.is_dir():
                p.mkdir(parents=True, exist_ok=True)
            return p
        p = Path(self.outputs_dir) / f"run_mv_adapter_{id(self)}"
        p.mkdir(parents=True, exist_ok=True)
        return p
