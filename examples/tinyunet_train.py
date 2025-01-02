"""
https://github.com/ChenJunren-Lab/TinyU-Net
"""

import torch
from dotenv import load_dotenv

from stem_cv.config import Models, Optimizers
from stem_cv.datasets import get_dataloaders
from stem_cv.trainers import train_simple
from stem_cv.tuners.learning_rate import find_lr

# Have a .env with:
# ===============
# MLFLOW_TRACKING_USERNAME=XXXXX
# MLFLOW_TRACKING_PASSWORD=XXXXX
# MLFLOW_S3_ENDPOINT_URL=XXXXX
# AWS_ACCESS_KEY_ID=XXXXX
# AWS_SECRET_ACCESS_KEY=XXXXX
# AWS_DEFAULT_REGION=XXXXX
# MLFLOW_TRACKING_URI=XXXXX
load_dotenv()

TEST_NAME = "test-schedule-free"

DO_LR_FIND = False

BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 5e-3

label_mapping = {
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
}


train_root = "./shared/study-hip-3dus-chop/huggingface/images/train"
val_root = "./shared/study-hip-3dus-chop/huggingface/images/val"

# Device
device = torch.device(2 if torch.cuda.is_available() else "cpu")

train_loader, val_loader = get_dataloaders(
    train_root,
    val_root,
    batch_size=BATCH_SIZE,
    label_mapping=label_mapping,
    train_pct=0.02,
)

if DO_LR_FIND:
    fig = find_lr(
        Models.TinyUnet,
        train_loader,
        device,
    )
    fig.savefig("lr_finder.png")
    exit(0)


train_simple(
    Models.TinyUnet,
    train_loader,
    val_loader,
    label_mapping,
    TEST_NAME,
    num_epochs=NUM_EPOCHS,
    device=device,
    lr=LEARNING_RATE,
    optimizer_name=Optimizers.AdamW,
)
