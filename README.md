<div align="center">

# 🌟 Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis

[![ICCV 2025](https://img.shields.io/badge/ICCV-2025-blue)](https://iccv2025.thecvf.com/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2411.17769)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://itsmag11.github.io/Omegance/)

**One Parameter, Infinite Possibilities** - Precise control over diffusion model detail granularity through the Omega parameter

[![Teaser Image](./figures/teaser.jpg)](https://itsmag11.github.io/Omegance/)

</div>

## 🎯 Project Overview

**Omegance** is a small tweak in the diffusion model that achieves precise control over image detail granularity through a single **Omega parameter**. Whether it's global, temporal (as in denoising process), or spatial effects, one parameter controls everything!

### ✨ Key Features

- 🎛️ **Single Parameter Control** - Control image details by simply adjusting the Omega value
- 🌍 **Global Granularity Control** - Influence the detail richness of the entire image
- ⏰ **Temporal Dynamic Scheduling** - Dynamically adjust detail control during generation
- 🗺️ **Spatial Regional Control** - Apply different detail control to different regions via masks
- 🔧 **Multi-Model Support** - Supports Stable Diffusion series, FLUX, Hunyuan, and more!

## 🚀 Quick Start

### Environment Setup

```bash
# Create conda environment
conda create --name omegance python=3.9
conda activate omegance

# Install PyTorch (CUDA 11.8)
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install diffusers==0.31.0 pytorch_lightning transformers==4.45.1 protobuf sentencepiece gradio
```

### 🎮 Interactive Demos

We provide three different Gradio demo interfaces:

#### 1. Global Effect Control
```bash
python gradio_global_sdxl.py
```
- Control the detail level of the entire image by adjusting the Omega value
- Positive values suppress details, negative values enhance details

#### 2. Spatial Regional Control
```bash
python gradio_controlnet_sdxl.py
```
- Use ControlNet conditions for spatial control
- Set different Omega values for different regions

#### 3. Sketch to Mask
```bash
python gradio_sketch2mask.py
```
- Convert user-drawn sketches to binary masks
- Prepare for subsequent spatial control

## 📖 Usage Guide

### Basic Usage

```python
import torch
from omegance_pipelines.pipeline_stable_diffusion_xl_snrcontrol import StableDiffusionXLSNRControlPipeline
from omegance_schedulers.scheduling_ddim_snrcontrol import DDIMSNRControlScheduler

# Load model
model_path = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMSNRControlScheduler.from_pretrained(model_path, subfolder="scheduler")
pipe = StableDiffusionXLSNRControlPipeline.from_pretrained(
    model_path, scheduler=scheduler, torch_dtype=torch.float16
).to("cuda")

# Generate image
prompt = "A beautiful landscape with mountains and lakes"
image = pipe(
    prompt=prompt,
    omega=10.0,  # Increase details
    num_inference_steps=50
).images[0]
```

### Advanced Usage

#### Temporal Dynamic Scheduling
```python
# Use predefined Omega scheduling strategies
# Uses: StableDiffusionXLSNRControlPipeline (same as global)
image = pipe(
    prompt=prompt,
    omega_schedule_type='exp1',  # Exponential scheduling
    num_inference_steps=50
).images[0]
```

#### Spatial Regional Control
```python
# Set different Omega values for different regions
# Uses: StableDiffusionXLControlNetSNRControlPipeline (different from global)
from omegance_pipelines.pipeline_controlnet_sd_xl_snrcontrol import StableDiffusionXLControlNetSNRControlPipeline
from diffusers import ControlNetModel

# Load ControlNet for spatial control
controlnet = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")
pipe_spatial = StableDiffusionXLControlNetSNRControlPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    scheduler=scheduler
)

image = pipe_spatial(
    prompt=prompt,
    image=control_image,  # ControlNet input image
    omega_mask=omega_mask,  # Spatial Omega mask
    controlnet_conditioning_scale=0.5
).images[0]
```

## 🔬 Technical Principles

### Pipeline Architecture

Omegance uses different pipeline classes for different control types:

| Control Type | Pipeline | Use Case | Key Parameters |
|--------------|----------|----------|----------------|
| **Global** | `StableDiffusionXLSNRControlPipeline` | Simple detail control | `omega` |
| **Temporal** | `StableDiffusionXLSNRControlPipeline` | Dynamic scheduling | `omega_schedule_type` |
| **Spatial** | `StableDiffusionXLControlNetSNRControlPipeline` | Regional control | `omega_mask` |

### How the Omega Parameter Works

The Omega parameter influences the diffusion process through:

1. **Noise Prediction Scaling**: `model_output = model_output * omega`
2. **Logistic Function Rescaling**: Maps user input Omega values to [0.95, 1.05] range
3. **Multi-Granularity Control**: Supports global, temporal, and spatial control

<!-- ### Supported Scheduling Strategies

- **EXP1/EXP2**: Exponential scheduling, suitable for progressive detail adjustment
- **COS1/COS2**: Cosine scheduling, suitable for smooth transitions
- **Custom Scheduling**: Supports user-defined timestep scheduling -->

<!-- ## 📊 Effect Showcase

### Global Effect Comparison

| Omega Value | Effect Description | Use Cases |
|-------------|-------------------|-----------|
| -5.0 | Minimalist style, least details | Abstract art, minimalist design |
| 0.0 | Original effect | Standard generation |
| 5.0 | Ultra-detailed, rich details | Fine illustrations, high-detail images |

### Temporal Dynamic Effects

Through different Omega scheduling strategies, you can achieve:
- Coarse early, fine later
- Fine early, coarse later
- Fine in the middle, coarse at both ends -->

## 🛠️ Inference Scripts

### Batch Generation Comparison

```bash
# Global effect comparison
bash sdxl-global_comparison.sh

# Temporal effect comparison
bash sdxl-temporal_comparison.sh

# Spatial effect comparison
bash sdxl-spatial_comparison.sh
```

<!-- ### Custom Inference

```bash
python sdxl_inference.py \
    --prompt "Your prompt here" \
    --omega 2.0 \
    --omega_schedule_type exp1 \
    --note "my_experiment"
``` -->

<!-- ## 📁 Project Structure

```
Omegance/
├── omegance_pipelines/          # Core pipeline implementations
│   ├── pipeline_stable_diffusion_xl_snrcontrol.py
│   ├── pipeline_flux_snrcontrol.py
│   └── utils/
├── omegance_schedulers/         # Scheduler implementations
│   ├── scheduling_ddim_snrcontrol.py
│   ├── scheduling_euler_discrete_snrcontrol.py
│   └── scheduling_flow_match_euler_discrete_snrcontrol.py
├── inference_scripts/           # Inference scripts
│   ├── sdxl_omega_schedule_inference.py
│   ├── flux_controlnet-canny_inference.py
│   └── ReNoise-Inversion/
├── gradio_*.py                  # Demo interfaces
└── figures/                     # Project images
```

## 🎨 Application Scenarios

- **Artistic Creation**: Control the detail level of painting styles
- **Product Design**: Adjust the fineness of product rendering
- **Content Generation**: Adjust image details according to needs
- **Style Transfer**: Achieve different granularity style conversions
- **Animation Production**: Control detail changes in animation frames -->

## 📚 Citation

If you use Omegance, please cite our paper:

```bibtex
@inproceedings{hou2025omegance,
  title={Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis},
  author={Hou, Xinyu and Yue, Zongsheng and Li, Xiaoming and Loy, Chen Change},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2025}
}
```
<!-- 
## 🤝 Contributing

We welcome contributions in all forms!

1. **Report Issues**: Report bugs or suggest improvements in Issues
2. **Submit Code**: Submit code improvements via Pull Requests
3. **Share Cases**: Share your use cases and effect demonstrations
4. **Improve Documentation**: Help improve documentation and tutorials -->

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgments

Thanks to the following open-source projects for support:
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
