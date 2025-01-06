import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import label

from stem_cv.datasets.process import postprocess_output, preprocess_image
from stem_cv.models.tiny_unet import load_model as load_tiny_unet
from stem_cv.models.utils import CMRF


class TinyBiUNet(nn.Module):
    def __init__(self, models):
        self.models = models

    def to(self, device):
        for model in self.models:
            model.to(device)

    def eval(self):
        for model in self.models:
            model.eval()

    def parameters(self):
        return self.models[0].parameters()

    def __call__(self, x):
        aggregated = None
        for i, model in enumerate(self.models):
            # Forward pass: Get raw outputs from the model
            output = model(x)  # Shape: (N, 4, H, W)

            # Apply argmax to get class predictions, removing redundant classes
            preds = torch.argmax(output, dim=1)  # Shape: (N, H, W)

            # One-hot encode the predictions
            one_hot = F.one_hot(preds, num_classes=4)  # Shape: (N, H, W, 4)
            one_hot = one_hot.permute(
                0, 3, 1, 2
            ).float()  # Shape: (N, 4, H, W)

            # Make the background class 0
            one_hot[:, 0] = 0
            # zero the orignal class
            if not i + 1 == 1:
                one_hot[:, i + 1] = one_hot[:, 1]
                one_hot[:, 1] = 0

            # Aggregate the one-hot encoded predictions
            if aggregated is None:
                aggregated = one_hot + 1
            else:
                aggregated += one_hot

        return aggregated


def load_model(model_paths, device="cpu", in_channels=1):
    models = []
    for model_path in model_paths:
        model = load_tiny_unet(
            model_path,
            device=device,
            in_channels=in_channels,
            num_classes=2,
        )
        models.append(model)

    return TinyBiUNet(models)


def infer(image_path, model_paths, label_mapping, device="cpu"):
    """
    Perform inference on a single image.
    """
    model = load_model(
        model_paths,
        device=device,
        in_channels=1,
    )

    image = Image.open(image_path).convert("RGB")
    # get image size
    size = image.size
    image_tensor = preprocess_image(image).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    pred_mask = postprocess_output(
        output,
        label_mapping,
    )

    # resize the mask to original image size
    pred_mask = Image.fromarray(pred_mask).resize(size)
    return pred_mask
