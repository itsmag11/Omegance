# !pip install opencv-python transformers accelerate
from diffusers import ControlNetModel
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
from controlnet_aux import OpenposeDetector

from torchvision.transforms import ToPILImage
from process.pose2mask import pose2mask

from omegance_pipelines.pipeline_controlnet_sd_xl_snrcontrol import StableDiffusionXLControlNetSNRControlPipeline
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

def get_args():
    parser = argparse.ArgumentParser(description="Text-to-Image argparse example")
    parser.add_argument('--prompt', '-p', type=str, help='prompt', default="A photo of a wedding")
    parser.add_argument('--negative_prompt', '-np', type=str, help='negative prompt', default="")
    
    ## scheduler with SNR control
    parser.add_argument('--omega', '-o', type=float, 
                        help='omega in DDIMSNRControlScheduler', 
                        default=0.0)
    parser.add_argument('--enable_omega_mask', '-om', action='store_true', help='enable automatic omega mask')
    parser.add_argument('--omega_in_black', '-oib', type=float, default=0.0)
    parser.add_argument('--omega_in_white', '-oiw', type=float, default=0.0)
    
    
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
        OUT_DIR = os.path.join(OUTPUT_BASE, 'sdxl-controlnet_pose', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'codebases'), exist_ok=True)
        args.out_dir = OUT_DIR

    # Initialize ControlNet pipeline.
    controlnet = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16)
    scheduler = DDIMSNRControlScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler",
                                                        omega_low=0.95, omega_high=1.05)
    pipe = StableDiffusionXLControlNetSNRControlPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, 
        scheduler=scheduler,
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()

    # -----------------------------------------------------------------------------------------------
    # Get control image
    # Compute openpose conditioning image.
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/person.png")
    control_image = openpose(image)
    control_image.save(os.path.join(OUT_DIR, 'inputs', 'pose.jpg'))

    extra_pipeline_kwargs = {}
    if args.enable_omega_mask:
        omega_mask = pose2mask(control_image)
        mask_pil = ToPILImage()(omega_mask)
        mask_pil.save(os.path.join(OUT_DIR, 'inputs', 'omega_mask.jpg'))

        omega_mask = np.array(omega_mask) / 255.0
        binary_mask = (omega_mask > 0.5).astype(np.uint8)
        
        omega_mask = np.where(binary_mask == 0, args.omega_in_black, args.omega_in_white)
        extra_pipeline_kwargs['omega_mask'] = omega_mask

    # -----------------------------------------------------------------------------------------------
    # Run text2image
    imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(RUN_TIMES):
        negative_prompt = "distorted lines, warped shapes, uneven grid patterns, irregular geometry, misaligned symmetry, low quality, bad quality, "
        # negative_prompt = "low quality, bad quality"
        image = pipe(
            args.prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=50,
            image=control_image.resize((1024, 1024)),
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