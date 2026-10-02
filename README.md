<div align="center">

# Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis

[Xinyu Hou](https://itsmag11.github.io/), [Zongsheng Yue](https://zsyoaoa.github.io/), [Xiaoming Li](https://csxmli2016.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

S-Lab, Nanyang Technological University

**ICCV 2025**

[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-4b44ce)](https://iccv2025.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2411.17769)
[![Project Page](https://img.shields.io/badge/Project-Page-4c9a2a)](https://itsmag11.github.io/Omegance/)
[![GitHub Stars](https://img.shields.io/github/stars/itsmag11/Omegance?style=social)](https://github.com/itsmag11/Omegance)

</div>

## 🎯 Project Overview

**Omegance** controls the detail granularity of diffusion-based synthesis with a single parameter, omega, without any
training. One line of code adds it to 🤗 Diffusers pipelines, for global, temporal or spatial (masked) control:
**omega < 0 gives more details, omega > 0 gives fewer details**, and omega = 0 leaves the output unchanged.

[![Teaser Image](./media/teaser.jpg)](https://itsmag11.github.io/Omegance/)

## 🚀 Quick Start

```bash
pip install git+https://github.com/itsmag11/Omegance
```

```python
import torch
from diffusers import StableDiffusionXLPipeline
from omegance import apply_omegance

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

apply_omegance(pipe, omega=-5)
image = pipe("a cozy wooden cabin in a snowy forest").images[0]
```

Tested with SDXL (incl. img2img, inpainting, ControlNet), AnimateDiff-SDXL, Latte, SD3 and FLUX (incl. ControlNet),
using the DDIM, Euler and Flow Matching schedulers. Other pipelines and schedulers may also work, but are untested.

## 🎛️ Usage

```python
# Global: useful range is about [-10, 10]
pipe.scheduler.set_omega(-5)
pipe.scheduler.set_omega(-5, active_range=(0.0, 0.5))  # only the early half of denoising (large timesteps)

# Temporal: presets "exp1", "exp2", "cos1", "cos2", or a list of per-step scales around 1.0
pipe.scheduler.set_omega(schedule="exp2")

# Spatial: black region -> low, white region -> high
from omegance import omega_mask
pipe.scheduler.set_omega(omega_mask(mask_image, low=10, high=-10))

# Turn off
from omegance import remove_omegance
remove_omegance(pipe)
```

## 🎮 Local Demo

An interactive Gradio demo with three tabs, each showing the result next to the omega = 0 baseline:

- **Global**: pick one omega for the whole image.
- **Temporal**: pick an omega schedule preset.
- **Spatial**: paint a region on the canvas and set one omega inside it and another outside.

```bash
git clone https://github.com/itsmag11/Omegance.git
cd Omegance
pip install -e ".[demo]"
python app.py
```

Then open http://127.0.0.1:7860 in your browser. The first run downloads SDXL (~7 GB).

| Option | Description |
|--------|-------------|
| `--model` | Another SDXL-compatible model, e.g. `--model SG161222/RealVisXL_V5.0` |
| `--offload` | CPU offload for GPUs with ~8 GB of memory (slower) |
| `--share` | Create a temporary public link to share the demo |
| `--port` | Server port (default `7860`) |

A CUDA GPU with ≥12 GB memory is recommended. Apple Silicon (MPS) and CPU also work, but are much slower.

## 📖 Citation

If you find our work useful for your research, please consider citing:

```bibtex
@inproceedings{hou2025omegance,
  title     = {Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis},
  author    = {Hou, Xinyu and Yue, Zongsheng and Li, Xiaoming and Loy, Chen Change},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```

## 📜 License

This project is licensed under the [S-Lab License 1.0](LICENSE). The diffusion models you use with Omegance (e.g.
Stable Diffusion XL, FLUX.1) are subject to their own licenses.
