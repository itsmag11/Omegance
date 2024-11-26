import gradio as gr
import numpy as np
import cv2
import os

## Code largly borrowed from
## https://github.com/lllyasviel/ControlNet/blob/main/gradio_scribble2image_interactive.py

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


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def create_canvas(w, h):
    return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

def process(input_image):
    # img = resize_image(HWC3(input_image['mask'][:, :, 0]), image_resolution)
    img = HWC3(input_image['layers'][0][:, :, 0])
    H, W, C = img.shape

    detected_map = np.zeros_like(img, dtype=np.uint8)
    detected_map[np.min(img, axis=2) > 127] = 255

    binary_mask = 255 - detected_map
    return binary_mask

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Interactive Scribbles")
    with gr.Column():
        with gr.Column():
            # input_image = gr.Image(source='upload', type='numpy', tool='sketch')
            input_image = gr.ImageEditor(sources=['upload'], type='numpy', brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"))
                    
            gr.Markdown(value='Do not forget to change your brush width to make it thinner. (Gradio do not allow developers to set brush width so you need to do it manually.) '
                              'Just click on the small pencil icon in the upper right corner of the above block.')
            run_button = gr.Button()

        binary_mask = gr.Image(type="numpy")
    ips = [input_image]
    run_button.click(fn=process, inputs=ips, outputs=[binary_mask])


block.launch(share=True)