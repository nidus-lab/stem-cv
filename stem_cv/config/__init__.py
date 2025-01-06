from enum import Enum

from schedulefree import AdamWScheduleFree
from segmentation_models_pytorch import Unet
from torch.optim import AdamW

from stem_cv.models.tiny_unet import TinyUNet


class Models(Enum):
    TinyUnet = "tiny_unet"
    UNetResnet34 = "unet_resnet34"
    UNetEfficientnet_b0 = "unet_efficientnet-b0"


class Optimizers(Enum):
    AdamWScheduleFree = "adam_w_schedule_free"
    AdamW = "adam_w"


def build_optimizer_from_name(name, model, lr):
    if name == Optimizers.AdamWScheduleFree:
        return AdamWScheduleFree(model.parameters(), lr), True
    elif name == Optimizers.AdamW:
        return AdamW(model.parameters(), lr), False
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def build_model_from_name(name, **kwargs):
    if name == Models.TinyUnet:
        return TinyUNet(**kwargs)
    elif name == Models.UNetResnet34:
        # change kwarg num_classes to classes
        kwargs["classes"] = kwargs.pop("num_classes")
        return Unet(**kwargs, encoder_name="resnet34")
    elif name == Models.UNetEfficientnet_b0:
        # change kwarg num_classes to classes
        kwargs["classes"] = kwargs.pop("num_classes")
        return Unet(**kwargs, encoder_name="efficientnet-b0")
    else:
        raise ValueError(f"Unknown model name: {name}")
