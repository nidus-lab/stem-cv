# MIT License
# Copyright (c) 2024 Chen Junren
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F

from stem_cv.datasets.process import postprocess_output, preprocess_image
from stem_cv.models.utils import CMRF, Conv, DWConv


class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.cmrf = CMRF(in_channels, out_channels)
        self.upsample = F.interpolate

    def forward(self, x, skip_connection):
        x = self.upsample(
            x, scale_factor=2, mode="bicubic", align_corners=False
        )
        x = torch.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x


class TinyUNet(nn.Module):
    """TinyU-Net with args(in_channels, num_classes)."""

    """
    in_channels: The number of input channels
    num_classes: The number of segmentation classes
    """

    def __init__(self, in_channels=3, num_classes=2):
        super(TinyUNet, self).__init__()
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]

        self.encoder1 = UNetEncoder(in_channels, 64)
        self.encoder2 = UNetEncoder(64, 128)
        self.encoder3 = UNetEncoder(128, 256)
        self.encoder4 = UNetEncoder(256, 512)

        self.decoder4 = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3 = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2 = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1 = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        x = self.final_conv(x)
        return x


def load_model(model_path, in_channels=3, num_classes=2, device="cpu"):
    """
    Load the trained model for inference.
    """
    model = TinyUNet(in_channels=in_channels, num_classes=num_classes)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()
    return model


def infer(image_path, model_path, label_mapping, device="cpu"):
    """
    Perform inference on a single image.
    """
    model = load_model(
        model_path, device=device, num_classes=len(label_mapping)
    )

    image_tensor = preprocess_image(image_path).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    pred_mask = postprocess_output(
        output,
        label_mapping,
    )
    return pred_mask
