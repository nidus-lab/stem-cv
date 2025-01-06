import os

import torch
import torch.nn.functional as F

from stem_cv.config import Optimizers, build_optimizer_from_name
from stem_cv.datasets import get_dataloaders
from stem_cv.models.tiny_bi_unet import load_model as load_bi_unet
from stem_cv.models.tiny_unet import load_model
from stem_cv.trainers import evaluate

BATCH_SIZE = 4

TEST_NAME = "test-baseline2"

MULTI = False

label_mapping = {
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
}
num_classes = len(label_mapping) + 1

train_root = (
    "./shared/study-hip-mixed-public-data/ultrasound/huggingface/images/train"
)
# val_root = (
#     "./shared/study-hip-mixed-public-data/ultrasound/huggingface/images/val"
# )
# train_root = "./shared/study-hip-3dus-chop/huggingface/images/train"
val_root = "./shared/study-hip-3dus-chop/huggingface/images/val"

# Device
device = torch.device(2 if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders(
    train_root,
    val_root,
    batch_size=BATCH_SIZE,
    label_mapping=label_mapping,
    train_pct=1,
    greyscale=True,
    num_workers=4,
)


def get_model_path(name):
    weight_dir = f"./stemcv_experiments/{name}/weights"
    for file in os.listdir(weight_dir):
        if "best_tiny_unet" in file:
            model_path = os.path.join(weight_dir, file)
            break

    return model_path


model_paths = []
if MULTI:
    for label in ["label1", "label2", "label3"]:
        model_path = get_model_path(f"{TEST_NAME}-{label}")
        model_paths.append(model_path)

    model = load_bi_unet(
        model_paths,
        device=device,
        in_channels=1,
    )

else:
    model_path = get_model_path(TEST_NAME)

    model = load_model(
        model_path,
        device=device,
        in_channels=1,
        num_classes=len(label_mapping) + 1,
    )

optimizer, opt_needs_set = build_optimizer_from_name(
    Optimizers.AdamWScheduleFree, model, lr=0.05
)

val_iou, val_dice, mean_val_loss, img = evaluate(
    model,
    optimizer,
    val_loader,
    label_mapping,
    device,
    opt_needs_set,
    num_classes=num_classes,
)

print(f"Validation IOU: {val_iou}")
print(f"Validation Dice: {val_dice}")

# save img
img.save("evaluated_mask.png")
