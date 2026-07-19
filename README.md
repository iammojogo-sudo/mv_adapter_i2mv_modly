# MV-Adapter (Multiview)

A Modly extension that generates 6 orthographic reference views (front, right, back, left, top, bottom) from a single input image using [MV-Adapter I2MV-SDXL](https://github.com/huanngzh/MV-Adapter) at 768×768 resolution.

## Pipeline

1. **Install** — `setup.py` creates an isolated venv, installs dependencies, and downloads the SDXL base model
2. **Download weights** — Click "install weights" in Modly to fetch the MV-Adapter adapter weight (~3.4 GB)
3. **Generate** — The first run generates 6 views + a 1×6 grid image, ready to feed into a 3D mesh generator

## Files

| File | Purpose |
|------|---------|
| `setup.py` | Venv creation, dependency install, SDXL base download |
| `generator.py` | Modly generator — runs bridge in subprocess |
| `bridge.py` | MV-Adapter i2mv SDXL pipeline (Plucker camera embeddings) |
| `manifest.json` | Modly extension manifest |

## Credits

- [MV-Adapter](https://github.com/huanngzh/MV-Adapter) by Zehuan Huang et al. (ICCV 2025)
- [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) by Stability AI
