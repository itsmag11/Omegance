import torch
from PIL import Image

from src.eunms import Model_Type, Scheduler_Type
from src.utils.enums_utils import get_pipes
from src.config import RunConfig

from main import run as invert

import argparse
from pytorch_lightning import seed_everything
import os
from datetime import datetime
import shutil
from diffusers.utils import load_image
import numpy as np

SEED = 666
OUTPUT_BASE = "results"
PATH_TO_IMAGE = ""
PATH_TO_MASK = ""

def write_args_to_file(args, file_name="args_output.txt"):
    with open(file_name, 'w') as file:
        for arg_name, arg_value in vars(args).items():  # Convert Namespace to a dict
            file.write(f"{arg_name}: {arg_value}\n")

def get_args():
    parser = argparse.ArgumentParser(description="Text-to-Image argparse example")
    ## scheduler with SNR control
    parser.add_argument('--omega', '-o', type=float, 
                        help='omega in DDIMSNRControlScheduler', 
                        default=0.0)
    parser.add_argument('--enable_omega_mask', '-om', action='store_true', help='enable omega mask')
    parser.add_argument('--omega_in_black', '-oib', type=float, default=0.0)
    parser.add_argument('--omega_in_white', '-oiw', type=float, default=0.0)
    
    parser.add_argument('--note', '-n', type=str, help='note', default="")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    args.seed = SEED

    # -----------------------------------------------------------------------------------------------
    # Prepare input
    args.prompt = "a kitten in a basket"
    img_path = PATH_TO_IMAGE
    mask_path = PATH_TO_MASK
    input_image = Image.open(img_path).convert("RGB").resize((1024, 1024))

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
        OUT_DIR = os.path.join(OUTPUT_BASE, 'renoise', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'codebases'), exist_ok=True)
        args.out_dir = OUT_DIR

    # -----------------------------------------------------------------------------------------------
    # Establish running pipeline
    model_type = Model_Type.SDXL_SNRControl
    scheduler_type = Scheduler_Type.DDIM_SNRControl
    pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

    # -----------------------------------------------------------------------------------------------
    # Prepare mask
    input_image.save(os.path.join(OUT_DIR, 'input.jpg'))
    extra_pipeline_kwargs = {}
    if args.enable_omega_mask:
        omega_mask = load_image(mask_path).convert('L')
        omega_mask = omega_mask.resize(input_image.size)
        omega_mask.save(os.path.join(OUT_DIR, 'omega_mask.jpg'))
        omega_mask = np.array(omega_mask) / 255.0
        binary_mask = (omega_mask > 0.5).astype(np.uint8)
        
        omega_mask = np.where(binary_mask == 0, args.omega_in_black, args.omega_in_white)
        extra_pipeline_kwargs['omega_mask'] = omega_mask

    config = RunConfig(model_type = model_type,
                        num_inference_steps = 50,
                        num_inversion_steps = 50,
                        num_renoise_steps = 1,
                        scheduler_type = scheduler_type,
                        perform_noise_correction = False,
                        seed = 7865)

    _, inv_latent, _, all_latents = invert(input_image,
                                        args.prompt,
                                        config,
                                        pipe_inversion=pipe_inversion,
                                        pipe_inference=pipe_inference,
                                        do_reconstruction=False)

    rec_image = pipe_inference(image = inv_latent,
                            prompt = args.prompt,
                            denoising_start=0.0,
                            num_inference_steps = config.num_inference_steps,
                            guidance_scale = 1.0,
                            omega=args.omega,
                            **extra_pipeline_kwargs).images[0]

    img_path = os.path.join(OUT_DIR, f'{RUNID}_inverted.jpg')
    rec_image.save(img_path)

    try:
        args.omega_bef_rescale_max, args.omega_bef_rescale_min = torch.max(pipe_inference.scheduler.omega_bef_rescale), torch.min(pipe_inference.scheduler.omega_bef_rescale)
        args.omega_aft_rescale_max, args.omega_aft_rescale_min = torch.max(pipe_inference.scheduler.omega_aft_rescale), torch.min(pipe_inference.scheduler.omega_aft_rescale)
    except:
        args.omega_bef_rescale = pipe_inference.scheduler.omega_bef_rescale
        args.omega_aft_rescale = pipe_inference.scheduler.omega_aft_rescale
    write_args_to_file(args, os.path.join(OUT_DIR, 'args.txt'))