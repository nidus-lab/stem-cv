from enum import Enum

from schedulefree import AdamWScheduleFree
from torch.optim import AdamW

from stem_cv.models.tiny_unet import TinyUNet


class Models(Enum):
    TinyUnet = "tiny_unet"


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
    else:
        raise ValueError(f"Unknown model name: {name}")
