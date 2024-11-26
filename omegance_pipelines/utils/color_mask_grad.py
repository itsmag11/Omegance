import numpy as np
import os
from PIL import Image

def create_gradual_colored_mask(mask):
    # Define the colors as RGB arrays (normalized to 0-1 range)
    positive_rgb = np.array([78, 102, 145]) / 255  # #4E6691
    negative_rgb = np.array([184, 71, 77]) / 255   # #B8474D
    white_rgb = np.array([1, 1, 1])  # White color for zero values

    abs_max = np.abs(mask).max()
    print(abs_max)
    if abs_max == 0:  # Avoid division by zero if all values are zero
        abs_max = 1
    normalized_mask = mask / abs_max
    # print(np.min(normalized_mask), np.max(normalized_mask)) # -1.0 0.0
    # print(normalized_mask.shape) # (1024, 1024)
    # raise NotImplementedError

    colored_mask = np.zeros((*mask.shape, 3))

    # Apply colors based on normalized values
    for i in range(3):  # Iterate over RGB channels
        # For positive values, interpolate between white and positive color
        colored_mask[:, :, i] = np.where(
            normalized_mask > 0,
            white_rgb[i] + normalized_mask * (positive_rgb[i] - white_rgb[i]),
            white_rgb[i] - normalized_mask * (negative_rgb[i] - white_rgb[i])
        )

    # Convert color_image to uint8 format for saving or displaying as an image
    colored_mask = (colored_mask * 255).astype(np.uint8)

    # Convert to PIL Image and save
    img = Image.fromarray(colored_mask)
    return img

def run_color_mask_grad(outdir, min, max):
    image = Image.open(os.path.join(outdir, 'inputs', 'omega_mask.jpg'))
    mask = np.array(image) / 255.0

    min_val = min
    max_val = max
    # Map the tensor from range (0, 1) to range (min_val, max_val), the smaller value in omega mask has already been transformed to 0
    # so only need to make sure 0 --> min, 1 --> max
    mask = mask * (max_val - min_val) + min_val

    color_mask = create_gradual_colored_mask(mask,)
    color_mask.save(os.path.join(outdir, 'inputs', 'color_mask.jpg'))
