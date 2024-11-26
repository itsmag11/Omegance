import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

RAD = 10

def pose2mask(pose_image):
    # Convert the PIL image to a NumPy array
    image = np.array(pose_image)

    # Convert RGB to BGR (OpenCV uses BGR format by default)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to get a binary image
    _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Dilate the binary image to make the skeleton lines thicker
    kernel = np.ones((RAD, RAD), np.uint8)
    dilated_image = cv2.dilate(binary_image, kernel, iterations=5)

    # Find contours of the dilated image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask
    mask = np.zeros_like(binary_image)

    # Fill the contours on the mask to make the whole human body white
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    binary_mask = torch.from_numpy(mask)

    return binary_mask
