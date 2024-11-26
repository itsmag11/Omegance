import os
import torch
import argparse
from datetime import datetime
from torchvision import transforms
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from diffusers.utils.testing_utils import enable_full_determinism

from omegance_pipelines.pipeline_flux_controlnet_snrcontrol import FluxControlNetSNRControlPipeline
from omegance_schedulers.scheduling_flow_match_euler_discrete_snrcontrol import FlowMatchEulerDiscreteSNRControlScheduler

import torch
from diffusers.utils import load_image
from PIL import Image
from diffusers import FluxControlNetModel
import numpy as np
import cv2

SEED = 666
RUN_TIMES = 3
NUM_ROWS = 5
OUTPUT_BASE = "results"
PATH_TO_MASK = ""

def extract_canny(original_image):
    image = np.array(original_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def write_args_to_file(args, file_name="args_output.txt"):
    with open(file_name, 'w') as file:
        for arg_name, arg_value in vars(args).items():  # Convert Namespace to a dict
            file.write(f"{arg_name}: {arg_value}\n")

def get_args():
    parser = argparse.ArgumentParser(description="Text-to-Image argparse example")
    parser.add_argument('--prompt', '-p', type=str, help='prompt', default="A girl in city, 25 years old, cool, futuristic")
    
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
        RUN_NAME = f'{RUNID}-s{SEED}-{NAME}-{args.note}'
        OUT_DIR = os.path.join(OUTPUT_BASE, 'flux-controlnet_canny', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'inputs'), exist_ok=True)
        args.out_dir = OUT_DIR

    enable_full_determinism()
    controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    base_model = "black-forest-labs/FLUX.1-schnell"
    scheduler = FlowMatchEulerDiscreteSNRControlScheduler.from_pretrained(base_model, subfolder="scheduler")
    pipe = FluxControlNetSNRControlPipeline.from_pretrained(base_model, 
                                                            scheduler=scheduler,
                                                            controlnet=controlnet, 
                                                            torch_dtype=torch.bfloat16).to(device)
    pipe.enable_model_cpu_offload()

    original_image = load_image("https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png")
    control_image = extract_canny(original_image)
    original_image.save(os.path.join(OUT_DIR, 'inputs', 'original_image.jpg'))
    control_image.save(os.path.join(OUT_DIR, 'inputs', 'control_image.jpg'))

    extra_pipeline_kwargs = {}
    if args.enable_omega_mask:
        omega_mask = load_image(PATH_TO_MASK).convert('L')
        omega_mask = omega_mask.resize(control_image.size)
        omega_mask.save(os.path.join(OUT_DIR, 'inputs', 'omega_mask.jpg'))
        omega_mask = np.array(omega_mask) / 255.0
        binary_mask = (omega_mask > 0.5).astype(np.uint8)
        
        omega_mask = np.where(binary_mask == 0, args.omega_in_black, args.omega_in_white)
        extra_pipeline_kwargs['omega_mask'] = omega_mask
        # raise NotImplementedError

    # -----------------------------------------------------------------------------------------------
    # Run text2image
    imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(RUN_TIMES):
        image = pipe(
            args.prompt,
            control_image=control_image,
            controlnet_conditioning_scale=0.6,
            num_inference_steps=4,
            guidance_scale=3.5,
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
