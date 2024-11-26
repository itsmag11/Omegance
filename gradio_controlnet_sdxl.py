import gradio as gr
from pytorch_lightning import seed_everything

from diffusers.utils import load_image
from diffusers import ControlNetModel, AutoencoderKL

import numpy as np
import cv2
from PIL import Image

from omegance_pipelines.pipeline_controlnet_sd_xl_snrcontrol import StableDiffusionXLControlNetSNRControlPipeline
from omegance_schedulers.scheduling_ddim_snrcontrol import DDIMSNRControlScheduler

import torch
torch.cuda.is_available()
torch.cuda.empty_cache()

with torch.no_grad(): 
    torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
controlnet_conditioning_scale = 0.5  # recommended for good generalization
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
scheduler = DDIMSNRControlScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler",
                                                        omega_low=0.95, omega_high=1.05)
pipe = StableDiffusionXLControlNetSNRControlPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    controlnet=controlnet, 
    vae=vae, 
    scheduler=scheduler,
    torch_dtype=torch.float16
)
pipe.enable_model_cpu_offload()

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def process(input_image):
    # print(input_image.keys())
    # print(input_image['background'].shape) ## (1024, 1024, 4) 
    # print(input_image['layers'][0].shape) ## (1024, 1024, 4)
    # print(input_image['composite'].shape) ## (1024, 1024, 4)
    
    # im = Image.fromarray(input_image['layers'][0]).convert("RGB")
    # im.save('layers.jpg')
    # im = Image.fromarray(input_image['composite']).convert("RGB")
    # im.save('composite.jpg')
    # raise NotImplementedError
    # img = resize_image(HWC3(input_image['mask'][:, :, 0]), image_resolution)
    img = HWC3(input_image['layers'][0][:, :, 0])
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) > 127] = 255

    binary_mask = 255 - detected_map
    return binary_mask

def generate_image(image, mask, black_value, white_value, prompt, seed):
    seed_everything(seed)
    # print(image)
    # # print(image['image'].shape)
    # # print(mask.shape)
    # raise NotImplementedError

    ## Get canny control image
    image = image['background'][:, :, :3]
    # print(image.shape) # (1024, 1024, 3)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)

    mask = mask[:, :, 0]
    mask = np.array(mask) / 255.0
    binary_mask = (mask > 0.5).astype(np.uint8)
    omega_mask = np.where(binary_mask == 0, black_value, white_value).astype(np.float64)
    # print(omega_mask)
    # raise NotImplementedError

    # negative_prompt = "distorted lines, warped shapes, uneven grid patterns, irregular geometry, misaligned symmetry, low quality, bad quality"
    image = pipe(prompt, 
            controlnet_conditioning_scale=controlnet_conditioning_scale, 
            image=control_image,
            omega_mask=omega_mask
        ).images[0]
    
    # Return generated and resized image
    return image


block = gr.Blocks().queue()
with block:
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Omegance with Interactive Omega Mask")
            with gr.Column():
                with gr.Row():
                    # input_image = gr.Image(source='upload', type='numpy', tool='sketch')
                    input_image = gr.ImageEditor(sources=['upload'], type='numpy', brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))
                    binary_mask = gr.Image(type="numpy")
                
                run_button = gr.Button()
                ips = [input_image]
                run_button.click(fn=process, inputs=ips, outputs=[binary_mask])

            inputs=[
                input_image,
                binary_mask,
                gr.Slider(label="Set Omega Value in Black Region (Before Rescale)", minimum=-10, maximum=10),
                gr.Slider(label="Set Omega Value in White Region (Before Rescale)", minimum=-10, maximum=10),
                gr.Textbox(label="Text Prompt", placeholder="Type your prompt here..."),
                gr.Number(label="Set Seed Number")
            ]
            run_button_t2i = gr.Button()
        out = gr.Image(type='pil')
        run_button_t2i.click(fn=generate_image, inputs=inputs, outputs=[out])

block.launch(share=True)