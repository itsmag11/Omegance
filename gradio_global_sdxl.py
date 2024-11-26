import gradio as gr
from pytorch_lightning import seed_everything

from omegance_pipelines.pipeline_stable_diffusion_xl_snrcontrol import StableDiffusionXLSNRControlPipeline
from omegance_schedulers.scheduling_euler_discrete_snrcontrol import EulerDiscreteSNRControlScheduler
from omegance_schedulers.scheduling_ddim_snrcontrol import DDIMSNRControlScheduler

import torch
torch.cuda.is_available()
torch.cuda.empty_cache()

with torch.no_grad(): 
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
# Create diffusion pipeline, load safetensors model file from project folder (from subfolders: "./"dreamlike-diffusion-1.0.safetensors")
model_path = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMSNRControlScheduler.from_pretrained(model_path, subfolder="scheduler")
pipe = StableDiffusionXLSNRControlPipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)

def generate_image(prompt, seed, omega):
    seed_everything(seed)

    negative_prompt = "distorted lines, warped shapes, uneven grid patterns, irregular geometry, misaligned symmetry, low quality, bad quality"
    image = pipe(prompt, negative_prompt=negative_prompt,
                 omega=omega).images[0]
    
    # Return generated and resized image
    return image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Omegance Global Effect")
            inputs=[
                gr.Textbox(label="Text Prompt", placeholder="Type your prompt here..."),
                gr.Number(label="Set Seed Number"),
                gr.Slider(label="Set Omega Value (Before Rescale)", minimum=-10.0, maximum=10.0),
            ]
            run_button_t2i = gr.Button()
        out = gr.Image(label="Generated Image")
        run_button_t2i.click(fn=generate_image, inputs=inputs, outputs=[out])

block.launch(share=True)
