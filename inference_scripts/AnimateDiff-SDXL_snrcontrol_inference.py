import torch
from diffusers.models import MotionAdapter
from diffusers.utils import export_to_gif

import os
import argparse
from datetime import datetime
from pytorch_lightning import seed_everything

from omegance_schedulers.scheduling_ddim_snrcontrol import DDIMSNRControlScheduler
from omegance_pipelines.pipeline_animatediff_sdxl import AnimateDiffSDXLSNRControlPipeline

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
    parser.add_argument('--prompt', '-p', type=str, help='prompt', default="a panda surfing in the ocean, realistic, high quality")
    
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
        RUN_NAME = f'{RUNID}-{NAME}-{args.note}'
        OUT_DIR = os.path.join(OUTPUT_BASE, 'animatediff', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'codebases'), exist_ok=True)
        args.out_dir = OUT_DIR

    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = DDIMSNRControlScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
        omega_low=0.95,
        omega_high=1.05
    )
    pipe = AnimateDiffSDXLSNRControlPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to(device)

    # enable memory savings
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    for i in range(RUN_TIMES):
        output = pipe(
            prompt=args.prompt,
            negative_prompt="low quality, worst quality",
            num_inference_steps=20,
            guidance_scale=8,
            width=1024,
            height=1024,
            num_frames=16,
            omega=args.omega
        )

        frames = output.frames[0]
        export_to_gif(frames, os.path.join(OUT_DIR, 'videos', f'{RUNID}_{str(i).zfill(3)}.gif'))

    try:
        args.omega_bef_rescale_max, args.omega_bef_rescale_min = torch.max(pipe.scheduler.omega_bef_rescale), torch.min(pipe.scheduler.omega_bef_rescale)
        args.omega_aft_rescale_max, args.omega_aft_rescale_min = torch.max(pipe.scheduler.omega_aft_rescale), torch.min(pipe.scheduler.omega_aft_rescale)
    except:
        args.omega_bef_rescale = pipe.scheduler.omega_bef_rescale
        args.omega_aft_rescale = pipe.scheduler.omega_aft_rescale
    args.omega_low = pipe.scheduler.omega_low
    args.omega_high = pipe.scheduler.omega_high
    write_args_to_file(args, os.path.join(OUT_DIR, 'args.txt'))