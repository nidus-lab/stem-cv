import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess the input image for inference.
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size)
    image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension
    # apply z-normalization using this images mean and std
    mean = image_tensor.mean()
    std = image_tensor.std()
    image_tensor = (image_tensor - mean) / std
    return image_tensor


def postprocess_output(output, label_mapping, needs_argmax=True):
    """
    Postprocess the model output into a segmentation map.
    """
    if needs_argmax:
        pred_mask = output.argmax(dim=1).squeeze(0)  # Remove batch dimension
    else:
        pred_mask = output.squeeze(0)

    reverse_label_mapping = {
        label: color for color, label in label_mapping.items()
    }

    # Create a new color image
    height, width = pred_mask.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Map labels back to colors
    for label, color in reverse_label_mapping.items():
        tmask = pred_mask == label
        tmask = tmask.cpu()
        color_image[tmask] = color

    return color_image
