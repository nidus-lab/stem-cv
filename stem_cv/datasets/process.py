import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Normalize


def keep_largest_components(pred, num_classes):
    """
    For each image in the batch, keep only the largest connected component
    for each predicted class, and remove all smaller components.

    Args:
        pred (torch.Tensor): Predicted segmentation map of shape (B, C, H, W).
        num_classes (int)  : Number of classes.

    Returns:
        torch.Tensor: A new prediction of shape (B, H, W) with only the
                      largest connected components for each class.
    """

    def process_image(pred_labels):
        cleaned_labels = np.zeros_like(pred_labels, dtype=np.int32)
        for c in range(num_classes):
            binary_map = (pred_labels == c).astype(np.uint8)
            labeled_map, num_components = label(binary_map)
            if num_components:
                largest_component = max(
                    range(1, num_components + 1),
                    key=lambda x: np.sum(labeled_map == x),
                )
                cleaned_labels[labeled_map == largest_component] = c
        return cleaned_labels

    outputs = [
        torch.from_numpy(
            process_image(torch.argmax(pred[b], dim=0).cpu().numpy())
        ).to(pred.device)
        for b in range(pred.shape[0])
    ]

    return torch.stack(outputs, dim=0)


def preprocess_image(image, target_size=(256, 256)):
    """
    Preprocess the input image for inference.
    """
    # make greyscale by taking only first channel
    image = np.array(image)[:, :, 0]
    image = Image.fromarray(image).resize(target_size)
    image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension
    # apply z-normalization using this images mean and std
    image_tensor = Normalize(mean=[0], std=[1])(image_tensor)
    return image_tensor


def postprocess_output(output, label_mapping, needs_argmax=True):
    """
    Postprocess the model output into a segmentation map.
    """
    if needs_argmax:
        pred_mask = keep_largest_components(
            output, len(label_mapping)
        ).squeeze(0)
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
        if label == 0:
            continue
        tmask = pred_mask == label
        tmask = tmask.cpu()
        color_image[tmask] = color

    return color_image
