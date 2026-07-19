"""
MV-Adapter — extension setup script.

Creates an isolated venv and installs all required dependencies.
Called by Modly at extension install time with:

    python setup.py <json_args>

where json_args contains:
    python_exe   — path to Modly's embedded Python (used to create the venv)
    ext_dir      — absolute path to this extension directory
    gpu_sm       — GPU compute capability as integer (0 on macOS)
    cuda_version — CUDA major/minor encoded as integer (e.g. 124, 128)
    torch_flavor — Flavor of torch to use (cuda, rocm - defaults to cuda)
    accelerator  — "mps" | "cuda" | "cpu" (passed by Electron since Modly 1.x)
    platform     — Electron's process.platform string ("win32", "darwin", "linux")
"""
import json
import platform
import subprocess
import sys
from pathlib import Path


def pip(venv: Path, *args: str) -> None:


def venv_python(venv: Path) -> Path:
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def setup(
    python_exe:    str,
    ext_dir:       Path,
    gpu_sm:        int,
    cuda_version:  int = 0,
    torch_flavor:  str = "cuda",
    accelerator:   str = "",
    platform_name: str = "",
    model_dir:     str = "",
    **_extra,
) -> None:
    venv = ext_dir / "venv"
    is_win = platform.system() == "Windows"
    is_mac = platform.system() == "Darwin" or platform_name == "darwin"
    machine = platform.machine().lower()
    is_linux_arm64 = platform.system() == "Linux" and machine in {"aarch64", "arm64"}

    if not accelerator:
        if is_mac:
            accelerator = "mps" if machine == "arm64" else "cpu"
        elif gpu_sm > 0:
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    print(f"[setup] accelerator={accelerator}  gpu_sm={gpu_sm}  cuda_version={cuda_version}")
    print(f"[setup] Creating venv at {venv} ...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ---- PyTorch ----
    if is_mac:
        print("[setup] macOS -> PyTorch from standard PyPI")
        pip(venv, "install", "torch", "torchvision")
    elif torch_flavor == "rocm":
        if is_win:
            print("[setup] WARNING: ROCm not on Windows. Falling back to CPU PyTorch.")
            pip(venv, "install", "torch==2.6.0", "torchvision==0.21.0",
                 "--index-url", "https://download.pytorch.org/whl/cpu")
        else:
            pip(venv, "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/rocm7.2")
    elif gpu_sm >= 100 or cuda_version >= 128:
        print(f"[setup] CUDA 12.8 path (sm{gpu_sm})")
        pip(venv, "install", "torch==2.7.0", "torchvision==0.22.0",
             "--index-url", "https://download.pytorch.org/whl/cu128")
    elif gpu_sm >= 70:
        print(f"[setup] CUDA 12.4 path (sm{gpu_sm})")
        pip(venv, "install", "torch==2.6.0", "torchvision==0.21.0",
             "--index-url", "https://download.pytorch.org/whl/cu124")
    else:
        print(f"[setup] CUDA 11.8 path (sm{gpu_sm})")
        pip(venv, "install", "torch==2.5.1", "torchvision==0.20.1",
             "--index-url", "https://download.pytorch.org/whl/cu118")

    # ---- Core dependencies ----
    print("[setup] Installing dependencies ...")
    pip(venv, "install",
        "numpy",
        "Pillow",
        "huggingface_hub>=0.20.0",
        "transformers",
        "diffusers",
        "safetensors",
        "einops",
        "tqdm",
        "scipy",
        "matplotlib",
        "jaxtyping",
        "typeguard",
        "trimesh",
        "rembg",
    )

    # ---- mvadapter (with its own pinned deps) ----
    print("[setup] Installing mvadapter ...")
    try:
        pip(venv, "install", "--no-deps", "mvadapter")
    except subprocess.CalledProcessError:
        print("[setup] WARNING: mvadapter installation failed. Check your internet connection.")
        print("[setup] MV-Adapter requires mvadapter package to function.")

    # ---- Download SDXL base model ----
    if model_dir:
        sdxl_dir = Path(model_dir) / "stable-diffusion-xl-base-1.0"
        if not (sdxl_dir / "model_index.json").exists():
            print("[setup] Downloading SDXL base model (~7 GB) ...")
            subprocess.run([
                str(venv_python(venv)), "-c",
                "from huggingface_hub import snapshot_download; import sys; "
                "snapshot_download("
                "repo_id='stabilityai/stable-diffusion-xl-base-1.0', "
                "local_dir=sys.argv[1], "
                "ignore_patterns=['*.md', 'LICENSE', 'NOTICE', '.gitattributes', '*.txt'], "
                "local_dir_use_symlinks=False)",
                str(sdxl_dir),
            ], check=True)
            print(f"[setup] SDXL base model downloaded to {sdxl_dir}")
        else:
            print(f"[setup] SDXL base model already present at {sdxl_dir}")

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            torch_flavor = sys.argv[5] if len(sys.argv) >= 6 else "cuda",
        )
    elif len(sys.argv) == 2:
        a = json.loads(sys.argv[1])
        setup(
            python_exe    = a["python_exe"],
            ext_dir       = Path(a["ext_dir"]),
            gpu_sm        = int(a.get("gpu_sm", 0)),
            cuda_version  = int(a.get("cuda_version", 0)),
            torch_flavor  = a.get("torch_flavor", "cuda"),
            accelerator   = a.get("accelerator", ""),
            platform_name = a.get("platform", ""),
            model_dir     = a.get("model_dir", ""),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version] [torch_flavor]")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":86}\'')
        sys.exit(1)
