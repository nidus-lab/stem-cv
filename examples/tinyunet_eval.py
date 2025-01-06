import os

import torch
import torch.nn.functional as F

from stem_cv.config import Optimizers, build_optimizer_from_name
from stem_cv.datasets import get_dataloaders
from stem_cv.models.tiny_unet import load_model
from stem_cv.trainers import evaluate

BATCH_SIZE = 4

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


class SuperModel:
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


opt_needs_set = False

TEST_NAME = "test-og-optstrongaugsv2-elastic-dice"
# TEST_NAME = "test-baseline2"

MULTI = False

if MULTI:
    models = []
    for label in ["label1", "label2", "label3"]:
        model_path = get_model_path(f"{TEST_NAME}-{label}")
        model = load_model(
            model_path,
            device=device,
            in_channels=1,
            num_classes=2,
        )
        models.append(model)

    model = SuperModel(models)

else:
    model_path = get_model_path(TEST_NAME)

    model = load_model(
        model_path,
        device=device,
        in_channels=1,
        num_classes=len(label_mapping) + 1,
    )

optimizer = build_optimizer_from_name(
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
