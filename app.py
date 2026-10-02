"""Local Gradio demo for Omegance.

    pip install "omegance[demo] @ git+https://github.com/itsmag11/Omegance"
    python app.py [--model MODEL] [--offload] [--share] [--port PORT]
"""

import argparse
from functools import partial

import gradio as gr
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler, EulerDiscreteScheduler

from omegance import PRESETS, apply_omegance, omega_mask

NEGATIVE = "low quality, bad quality, distorted, blurry"


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pipeline(model, offload=False, device=None):
    device = device or default_device()
    dtype = torch.float32 if device == "cpu" else torch.float16
    pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    if offload and device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.omegance_schedulers = {
        "ddim": apply_omegance(DDIMScheduler.from_config(pipe.scheduler.config)),
        "euler": apply_omegance(EulerDiscreteScheduler.from_config(pipe.scheduler.config)),
    }
    return pipe


def _generate(pipe, scheduler, prompt, seed, steps, size):
    pipe.scheduler = pipe.omegance_schedulers[scheduler]
    return pipe(
        prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=int(steps),
        height=int(size),
        width=int(size),
        generator=torch.Generator("cpu").manual_seed(int(seed)),
    ).images[0]


def run_global(pipe, prompt, seed, steps, size, omega):
    scheduler = pipe.omegance_schedulers["ddim"]
    scheduler.set_omega(0.0)
    reference = _generate(pipe, "ddim", prompt, seed, steps, size)
    scheduler.set_omega(omega)
    return reference, _generate(pipe, "ddim", prompt, seed, steps, size)


def run_temporal(pipe, prompt, seed, steps, size, preset):
    scheduler = pipe.omegance_schedulers["euler"]
    scheduler.set_omega(0.0)
    reference = _generate(pipe, "euler", prompt, seed, steps, size)
    scheduler.set_omega(schedule=preset)
    return reference, _generate(pipe, "euler", prompt, seed, steps, size)


def run_spatial(pipe, prompt, seed, steps, size, sketch, omega_outside, omega_inside):
    layers = (sketch or {}).get("layers") or []
    if not layers:
        raise gr.Error("Paint a region on the canvas first.")
    painted = np.max([np.asarray(layer)[..., -1] for layer in layers], axis=0) > 0
    scheduler = pipe.omegance_schedulers["ddim"]
    scheduler.set_omega(omega_mask(painted.astype(np.float32), low=omega_outside, high=omega_inside))
    image = _generate(pipe, "ddim", prompt, seed, steps, size)
    scheduler.set_omega(0.0)
    return (painted * 255).astype(np.uint8), image


def build_demo(pipe, default_size=1024):
    def common_inputs():
        prompt = gr.Textbox(label="Prompt", value="A cozy wooden cabin in a snowy forest, golden hour")
        with gr.Row():
            seed = gr.Number(label="Seed", value=0, precision=0)
            steps = gr.Slider(label="Steps", minimum=10, maximum=50, value=30, step=1)
            size = gr.Slider(label="Resolution", minimum=512, maximum=1536, value=default_size, step=64)
        return [prompt, seed, steps, size]

    with gr.Blocks(title="Omegance") as demo:
        gr.Markdown(
            "# 🌟 Omegance\n"
            "One parameter for detail granularity. **omega < 0: more details, omega > 0: fewer details.**"
        )
        with gr.Tab("Global"):
            inputs = common_inputs()
            omega = gr.Slider(label="Omega", minimum=-10, maximum=10, value=-5, step=0.5)
            button = gr.Button("Generate", variant="primary")
            with gr.Row():
                outputs = [gr.Image(label="omega = 0"), gr.Image(label="Omegance")]
            button.click(partial(run_global, pipe), inputs + [omega], outputs, api_name="global")

        with gr.Tab("Temporal"):
            inputs = common_inputs()
            preset = gr.Radio(sorted(PRESETS), label="Omega schedule", value="exp2")
            gr.Markdown(
                "`exp1`: rich details early, fading back to neutral · `exp2`: rich details early, smoother later · "
                "`cos1`: neutral early, smoother later · `cos2`: smooth early, more detailed later"
            )
            button = gr.Button("Generate", variant="primary")
            with gr.Row():
                outputs = [gr.Image(label="omega = 0"), gr.Image(label="Omegance")]
            button.click(partial(run_temporal, pipe), inputs + [preset], outputs, api_name="temporal")

        with gr.Tab("Spatial"):
            inputs = common_inputs()
            sketch = gr.ImageEditor(
                label="Paint the region that gets the second omega",
                value={"background": np.full((512, 512, 3), 255, dtype=np.uint8), "layers": [], "composite": None},
                brush=gr.Brush(colors=["#000000"], color_mode="fixed", default_size=48),
                type="numpy",
                image_mode="RGBA",
            )
            with gr.Row():
                omega_outside = gr.Slider(label="Omega outside painted region", minimum=-10, maximum=10, value=10, step=0.5)
                omega_inside = gr.Slider(label="Omega in painted region", minimum=-10, maximum=10, value=-10, step=0.5)
            button = gr.Button("Generate", variant="primary")
            with gr.Row():
                outputs = [gr.Image(label="Mask (white = painted)"), gr.Image(label="Omegance")]
            button.click(partial(run_spatial, pipe), inputs + [sketch, omega_outside, omega_inside], outputs,
                         api_name="spatial")
    return demo


def main():
    parser = argparse.ArgumentParser(description="Omegance local demo")
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Any SDXL-compatible text-to-image model on the Hub or a local path")
    parser.add_argument("--offload", action="store_true", help="CPU offload to fit GPUs with ~8 GB of memory")
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    pipe = load_pipeline(args.model, offload=args.offload)
    build_demo(pipe).queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
