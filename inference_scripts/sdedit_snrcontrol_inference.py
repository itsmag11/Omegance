import torch
from diffusers.utils import make_image_grid, load_image

import os
import torch
import argparse
from datetime import datetime
from torchvision import transforms
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
import numpy as np

from omegance_pipelines.pipeline_stable_diffusion_xl_img2img_snrcontrol import StableDiffusionXLImg2ImgSNRControlPipeline
from omegance_schedulers.scheduling_euler_discrete_snrcontrol import EulerDiscreteSNRControlScheduler


SEED = 666
RUN_TIMES = 5
NUM_ROWS = 5
OUTPUT_BASE = "results"
PATH_TO_MASK = ""


def write_args_to_file(args, file_name="args_output.txt"):
    with open(file_name, 'w') as file:
        for arg_name, arg_value in vars(args).items():  # Convert Namespace to a dict
            file.write(f"{arg_name}: {arg_value}\n")

def get_args():
    parser = argparse.ArgumentParser(description="Text-to-Image argparse example")
    parser.add_argument('--prompt', '-p', type=str, help='prompt', default="A photo of a wedding")
    
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
        RUN_NAME = f'{RUNID}-{NAME}-{args.note}'
        OUT_DIR = os.path.join(OUTPUT_BASE, 'sdedit', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'inputs'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'codebases'), exist_ok=True)
        args.out_dir = OUT_DIR

    scheduler = EulerDiscreteSNRControlScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")
    pipeline = StableDiffusionXLImg2ImgSNRControlPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        scheduler=scheduler,
        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.enable_model_cpu_offload()


    # prepare image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/baklava.jpg"
    init_image = load_image(url)
    init_image.save(os.path.join(OUT_DIR, 'inputs', 'input.jpg'))

    extra_pipeline_kwargs = {}
    if args.enable_omega_mask:
        omega_mask = load_image(PATH_TO_MASK).convert('L')
        omega_mask = omega_mask.resize(init_image.size)
        omega_mask.save(os.path.join(OUT_DIR, 'inputs', 'omega_mask.jpg'))
        omega_mask = np.array(omega_mask) / 255.0
        binary_mask = (omega_mask > 0.5).astype(np.uint8)
        
        omega_mask = np.where(binary_mask == 0, args.omega_in_black, args.omega_in_white)
        extra_pipeline_kwargs['omega_mask'] = omega_mask

    # -----------------------------------------------------------------------------------------------
    # Run text2image
    imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(RUN_TIMES):
        # pass prompt and image to pipeline
        image = pipeline(args.prompt,
                        negative_prompt="bad quality, low quality, distorted shape, distorted shape",
                        image=init_image, strength=0.8,
                        omega=args.omega,
                        omega_low=0.8,
                        omega_high=1.2,  
                        **extra_pipeline_kwargs).images[0]
        make_image_grid([init_image, image], rows=1, cols=2)

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
        args.omega_bef_rescale_max, args.omega_bef_rescale_min = torch.max(pipeline.scheduler.omega_bef_rescale), torch.min(pipeline.scheduler.omega_bef_rescale)
        args.omega_aft_rescale_max, args.omega_aft_rescale_min = torch.max(pipeline.scheduler.omega_aft_rescale), torch.min(pipeline.scheduler.omega_aft_rescale)
    except:
        args.omega_bef_rescale = pipeline.scheduler.omega_bef_rescale
        args.omega_aft_rescale = pipeline.scheduler.omega_aft_rescale
    write_args_to_file(args, os.path.join(OUT_DIR, 'args.txt'))