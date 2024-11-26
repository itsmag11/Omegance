<div align="center">
<h1>Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis</h1>

[Xinyu Hou](https://itsmag11.github.io/), [Zongsheng Yue](https://zsyoaoa.github.io/), [Xiaoming Li](https://csxmli2016.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)

<div>
    <sup></sup>S-Lab, Nanyang Technological University
</div>

[Paper]() | [Project Page](https://itsmag11.github.io/Omegance/)

<img src="./figures/teaser.jpg" width="800px">

</div>

## Installation
```
conda create --name omegance python=3.9
conda activate omegance
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install diffusers==0.31.0 pytorch_lightning transformers==4.45.1 protobuf sentencepiece
```

## Gradio Demo

We provide local gradio demo for you to easily interact with Omegance. First, we need to install ```gradio``` package to use the interface:
```
pip install gradio
```

For global control, run:
```
python gradio_global_sdxl.py
```

For spatial control using ControlNet-Canny with self-drawn masks, run:
```
python gradio_controlnet_sdxl.py
```

To generate binary mask from user-provided strokes for other tasks, run:
```
python gradio_sketch2mask.py
```

## Inference with Omegance

The inference codes for various applications using Omegance are avaliable. Results will be automatically saved to ```./results/``` directory.

### Global Effects

For comparisons on global effects of Omegance, run:
```
bash sdxl-global_comparison.sh
```
Three sets of results (detail-, original, detail+) will be generated. Compare them to see the global effects of granularity control.

### Temporal Effects

To compare temporal effects of different omega schedules (EXP1, EXP2, COS1, COS2 as in the paper), run:
```
bash sdxl-temporal_comparison.sh
```

### Spatial Effects

To demonstrate the effects of different omega masks, we use ControlNet-Depth as examples. Run:
```
bash sdxl-spatial_comparison.sh
```
