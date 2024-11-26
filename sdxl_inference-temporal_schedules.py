import os
import torch
import argparse
from datetime import datetime
from torchvision import transforms
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from diffusers.utils.testing_utils import enable_full_determinism

from omegance_pipelines.pipeline_stable_diffusion_xl_snrcontrol import StableDiffusionXLSNRControlPipeline
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
    
    ## scheduler with SNR control
    parser.add_argument('--omega', '-o', type=float, 
                        help='omega in DDIMSNRControlScheduler', 
                        default=0.0)
    parser.add_argument('--omega_start_step', '-oss', type=int, 
                        help='Inference step that omega scaling starts', 
                        default=0)
    parser.add_argument('--omega_end_step', '-oes', type=int, 
                        help='Inference step that omega scaling ends', 
                        default=-1)
    
    ## omega schedule (If set, o/oss/oes will be over-written)
    parser.add_argument('--omega_schedule_type', '-ost', type=str, choices=[None, 'exp1', 'exp2', 'cos1', 'cos2'],
                        help='Omega schedule type', 
                        default=None)
    
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
        OUT_DIR = os.path.join(OUTPUT_BASE, 'sdxl-temporal', RUN_NAME)
        os.makedirs(OUT_DIR, exist_ok=True)
        os.makedirs(os.path.join(OUT_DIR, 'images'), exist_ok=True)
        args.out_dir = OUT_DIR

    # -----------------------------------------------------------------------------------------------
    # Establish running pipeline
    enable_full_determinism()
    args.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    scheduler = EulerDiscreteSNRControlScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    scheduler.disable_omega_rescale()
    pipeline_text2image = StableDiffusionXLSNRControlPipeline.from_pretrained(
                        args.model_path,
                        scheduler=scheduler,
                        torch_dtype=torch.float16, variant="fp16", use_safetensors=True
                        ).to(device)

    # -----------------------------------------------------------------------------------------------
    # Run text2image
    imgs = []
    transform = transforms.Compose([transforms.ToTensor()])
    for i in range(RUN_TIMES):
        negative_prompt = "distorted lines, warped shapes, uneven grid patterns, irregular geometry, misaligned symmetry, low quality, bad quality"
        image = pipeline_text2image(prompt=PROMPT,
                                    negative_prompt=negative_prompt,
                                    omega=args.omega,
                                    omega_start_step=args.omega_start_step,
                                    omega_end_step=args.omega_end_step,
                                    omega_schedule_type=args.omega_schedule_type).images[0] ## PIL.Image.Image

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
        args.omega_bef_rescale_max, args.omega_bef_rescale_min = torch.max(pipeline_text2image.scheduler.omega_bef_rescale), torch.min(pipeline_text2image.scheduler.omega_bef_rescale)
        args.omega_aft_rescale_max, args.omega_aft_rescale_min = torch.max(pipeline_text2image.scheduler.omega_aft_rescale), torch.min(pipeline_text2image.scheduler.omega_aft_rescale)
    except:
        args.omega_bef_rescale = pipeline_text2image.scheduler.omega_bef_rescale
        args.omega_aft_rescale = pipeline_text2image.scheduler.omega_aft_rescale
    write_args_to_file(args, os.path.join(OUT_DIR, 'args.txt'))
