import os
import torch
import argparse
from datetime import datetime
from torchvision import transforms
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
import shutil
from diffusers.utils.testing_utils import enable_full_determinism
from diffusers.utils import load_image

from omegance_pipelines.pipeline_stable_diffusion_xl_inpaint_snrcontrol import StableDiffusionXLInpaintSNRControlPipeline
from omegance_schedulers.scheduling_euler_discrete_snrcontrol import EulerDiscreteSNRControlScheduler

SEED = 666
RUN_TIMES = 5
NUM_ROWS = 5
OUTPUT_BASE = "results"

def write_args_to_file(args, file_name="args_output.txt"):
    with open(file_name, 'w') as file:
        for arg_name, arg_value in vars(args).items():  # Convert Namespace to a dict
            file.write(f"{arg_name}: {arg_value}\n")

def get_args():
    parser = argparse.ArgumentParser(description="Text-to-Image argparse example")
    parser.add_argument('--prompt', '-p', type=str, help='prompt', default="concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k")
    
    ## scheduler with SNR control
    parser.add_argument('--omega', '-o', type=float, 
                        help='omega in DDIMSNRControlScheduler', 
                        default=0.0)
    
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
        OUT_DIR = os.path.join(OUTPUT_BASE, 'sdxl-inpainting', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'inputs'), exist_ok=True)
        args.out_dir = OUT_DIR

    # -----------------------------------------------------------------------------------------------
    # Establish running pipeline
    enable_full_determinism()
    scheduler = EulerDiscreteSNRControlScheduler.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", subfolder="scheduler")
    pipeline = StableDiffusionXLInpaintSNRControlPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
        scheduler=scheduler,
        torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()

    # -----------------------------------------------------------------------------------------------
    # Prepare model inputs
    # load base and mask image
    init_image = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/alex-iby-G_Pk4D9rMLs.png")
    mask_image = load_image("https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/alex-iby-G_Pk4D9rMLs_mask.png")
    
    init_image.save(os.path.join(OUT_DIR, 'inputs', 'init_image.jpg'))
    mask_image.save(os.path.join(OUT_DIR, 'inputs', 'mask_image.jpg'))

    # -----------------------------------------------------------------------------------------------
    # Run
    imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(RUN_TIMES):
        image = pipeline(prompt=args.prompt,
                         negative_prompt="low quality, bad quality, distorted shapes, distorted lines",
                         image=init_image, 
                         mask_image=mask_image,
                         omega=args.omega).images[0]

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