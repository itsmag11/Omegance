# Inference Scripts

This directory contains inference scripts for different applications using Omegance.

## 📋 Scripts by Application

### 🖼️ Text-to-Image (T2I)

| Model | Script |
|-------|--------|
| SDXL | [`../sdxl_inference.py`](../sdxl_inference.py) |
| SDXL Omega Schedule | [`sdxl_omega_schedule_inference.py`](sdxl_omega_schedule_inference.py) |
| Flux | [`flux_inference.py`](flux_inference.py) |
| Stable Diffusion 3 | [`sd3_snrcontrol_inference.py`](sd3_snrcontrol_inference.py) |
| RealVisXL V5.0 | [`RealVisXL_V5.0_snrcontrol_inference.py`](RealVisXL_V5.0_snrcontrol_inference.py) |

### 🎬 Text-to-Video (T2V)

| Model | Script |
|-------|--------|
| LaTte | [`latte_snrcontrol_inference.py`](latte_snrcontrol_inference.py) |
| AnimateDiff SDXL | [`AnimateDiff-SDXL_snrcontrol_inference.py`](AnimateDiff-SDXL_snrcontrol_inference.py) |

### 🎨 Image Editing

| Model | Script |
|-------|--------|
| SDXL Inpainting | [`sdxl_inpainting_snrcontrol_inference.py`](sdxl_inpainting_snrcontrol_inference.py) |
| SDEdit | [`sdedit_snrcontrol_inference.py`](sdedit_snrcontrol_inference.py) |

### 🎯 Spatial Control (ControlNet)

| Model | Script |
|-------|--------|
| SDXL + ControlNet Canny | [`controlnet/sdxl_controlnet_canny.py`](controlnet/sdxl_controlnet_canny.py) |
| SDXL + ControlNet Depth | [`controlnet/sdxl_controlnet_depth.py`](controlnet/sdxl_controlnet_depth.py) |
| SDXL + ControlNet Pose | [`controlnet/sdxl_controlnet_pose.py`](controlnet/sdxl_controlnet_pose.py) |
| Flux + ControlNet Canny | [`flux_controlnet-canny_inference.py`](flux_controlnet-canny_inference.py) |

### 🔄 Inversion

| Model | Script |
|-------|--------|
| ReNoise | [`ReNoise-Inversion/`](ReNoise-Inversion/) |

