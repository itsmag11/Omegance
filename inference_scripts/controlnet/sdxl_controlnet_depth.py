# pip install accelerate transformers safetensors diffusers

import torch
import numpy as np

from transformers import DPTImageProcessor, DPTForDepthEstimation

# !pip install opencv-python transformers accelerate
from diffusers import ControlNetModel, AutoencoderKL
from diffusers.utils import load_image
import numpy as np
import torch

import os
import argparse
import shutil
from torchvision import transforms
from datetime import datetime
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid

from PIL import Image

from omegance_pipelines.pipeline_controlnet_sd_xl_img2img import StableDiffusionXLControlNetImg2ImgSNRControlPipeline
from omegance_schedulers.scheduling_ddim_snrcontrol import DDIMSNRControlScheduler
from omegance_schedulers.scheduling_euler_discrete_snrcontrol import EulerDiscreteSNRControlScheduler


SEED = 666
RUN_TIMES = 3
NUM_ROWS = 5
OUTPUT_BASE = "results"


def write_args_to_file(args, file_name="args_output.txt"):
    with open(file_name, 'w') as file:
        for arg_name, arg_value in vars(args).items():  # Convert Namespace to a dict
            file.write(f"{arg_name}: {arg_value}\n")

def normalize_to_pil(tensor):
    min_val, max_val = torch.min(tensor), torch.max(tensor)

    # Step 1: Normalize to [0, 1]
    tensor_normalized = (tensor - min_val) / (max_val - min_val)
    
    # Step 2: Scale to [0, 255] and convert to uint8
    tensor_255 = (tensor_normalized * 255).clamp(0, 255).byte()
    
    # Step 3: Convert to PIL image and save
    pil_image = transforms.ToPILImage()(tensor_255.squeeze(0))
    return pil_image

def get_args():
    parser = argparse.ArgumentParser(description="Text-to-Image argparse example")
    parser.add_argument('--prompt', '-p', type=str, help='prompt', default="A robot, 4k photo")
    
    ## scheduler with SNR control
    parser.add_argument('--omega', '-o', type=float, 
                        help='omega in DDIMSNRControlScheduler', 
                        default=0.0)
    
    parser.add_argument('--enable_omega_mask', '-om', action='store_true', help='enable omega mask')
    parser.add_argument('--omega_in_back', '-oib', type=float, default=0.0)
    parser.add_argument('--omega_in_front', '-oif', type=float, default=0.0)
    
    parser.add_argument('--note', '-n', type=str, help='note', default="")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    PROMPT = args.prompt
    args.seed = SEED
    args.run_times = RUN_TIMES
    args.num_rows = NUM_ROWS

    # -----------------------------------------------------------------------------------------------
    # Setting running parameters and output dir
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR = None
    if args.note != 'debug':
        now = datetime.now()
        RUNID = now.strftime("%Y%m%d%H%M")
        NAME = args.prompt.replace(' ', '_')
        NAME = NAME[:30] if len(NAME) > 30 else NAME
        RUN_NAME = f'{RUNID}-{NAME}-{args.note}'
        OUT_DIR = os.path.join(OUTPUT_BASE, 'sdxl-controlnet_depth', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'codebases'), exist_ok=True)
        args.out_dir = OUT_DIR


    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0-small",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    scheduler = DDIMSNRControlScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    pipe = StableDiffusionXLControlNetImg2ImgSNRControlPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        scheduler=scheduler,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe.enable_model_cpu_offload()


    def get_depth_map(image):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)

        A = args.omega_in_back
        B = args.omega_in_front
        omega_mask = depth_map.squeeze(0).squeeze(0) * (B - A) + A
        print(omega_mask.shape) # torch.Size([1024, 1024])

        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image, omega_mask


    image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"
    ).resize((1024, 1024))
    image.save(os.path.join(OUT_DIR, 'inputs', 'input_image.jpg'))
    controlnet_conditioning_scale = 0.5  # recommended for good generalization
    control_image, omega_mask = get_depth_map(image)
    control_image.save(os.path.join(OUT_DIR, 'inputs', 'depth_image.jpg'))

    extra_pipeline_kwargs = {}
    if args.enable_omega_mask:
        omega_mask_pil = normalize_to_pil(omega_mask)
        omega_mask_pil.save(os.path.join(OUT_DIR, 'inputs', 'omega_mask.jpg'))
        extra_pipeline_kwargs['omega_mask'] = omega_mask

    # -----------------------------------------------------------------------------------------------
    # Run text2image
    imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(RUN_TIMES):
        image = pipe(
            args.prompt,
            image=image,
            control_image=control_image,
            strength=0.99,
            num_inference_steps=50,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            omega=args.omega,
            **extra_pipeline_kwargs
        ).images[0]

        img_path = os.path.join(OUT_DIR, 'images', f'{RUNID}_{str(i).zfill(3)}.jpg')
        image.save(img_path)

        imgs.append(transform(image))

    # -----------------------------------------------------------------------------------------------
    # Visualize results
    grid = make_grid(imgs, nrow=NUM_ROWS)
    transform_back = transforms.ToPILImage()
    grid_out = transform_back(grid)
    grid_out.save(os.path.join(OUT_DIR, 'grid.jpg'))

    try:
        args.omega_bef_rescale_max, args.omega_bef_rescale_min = torch.max(pipe.scheduler.omega_bef_rescale), torch.min(pipe.scheduler.omega_bef_rescale)
        args.omega_aft_rescale_max, args.omega_aft_rescale_min = torch.max(pipe.scheduler.omega_aft_rescale), torch.min(pipe.scheduler.omega_aft_rescale)
    except:
        args.omega_bef_rescale = pipe.scheduler.omega_bef_rescale
        args.omega_aft_rescale = pipe.scheduler.omega_aft_rescale
    write_args_to_file(args, os.path.join(OUT_DIR, 'args.txt'))