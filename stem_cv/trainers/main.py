import itertools
import os
import random
from sched import scheduler

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow import create_experiment, get_experiment_by_name, set_experiment
from PIL import Image
from torchmetrics.classification import Dice, MulticlassJaccardIndex

from stem_cv.config import (Optimizers, build_model_from_name,
                            build_optimizer_from_name)
from stem_cv.datasets import apply_image_denorm, display_augs
from stem_cv.datasets.process import postprocess_output

BUCKET_NAME = "stemcv-artifacts"


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    opt_needs_set,
    device,
    epoch,
    num_epochs,
    scheduler=None,
):
    model.train()
    if opt_needs_set:
        optimizer.train()
    running_loss = 0.0

    dice_metric = Dice(num_classes=dataloader.num_classes, ignore_index=0).to(
        device
    )

    # Get total number of images
    total_images = len(dataloader.dataset)

    for i, (images, masks) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)

        ce_loss = nn.CrossEntropyLoss()(outputs, masks)

        preds = outputs.argmax(dim=1)

        dsc_loss = 1 - dice_metric(preds, masks)
        dice_metric.reset()

        loss = ce_loss + dsc_loss
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

        if (i + 1) % int(total_images / 10) == 0 or i == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}"
            )

            mlflow.log_metric("train_loss", loss.item(), step=i)
            mlflow.log_metric("train_ce_loss", ce_loss.item(), step=i)
            mlflow.log_metric("train_dsc_loss", dsc_loss.item(), step=i)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# During evaluation
@torch.no_grad()
def evaluate(
    model,
    optimizer,
    dataloader,
    label_mapping,
    device,
    opt_needs_set,
    num_classes,
):
    iou_metric = MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=0
    ).to(device)
    dice_metric = Dice(num_classes=num_classes, ignore_index=0).to(device)

    model.eval()
    if opt_needs_set:
        optimizer.eval()
    iou_metric.reset()
    dice_metric.reset()

    rows = []

    ce_losses = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        r_idx = random.randint(0, len(images) - 1)

        outputs = model(images)
        ce_loss = nn.CrossEntropyLoss()(outputs, masks)

        pred_class = outputs.argmax(dim=1)  # Shape: (N, H, W)

        # Update metrics
        iou_metric.update(pred_class, masks)
        dice_metric.update(pred_class, masks)

        pred_mask = postprocess_output(
            pred_class[r_idx], label_mapping, needs_argmax=False
        )
        gt_mask = postprocess_output(
            masks[r_idx], label_mapping, needs_argmax=False
        )

        # Creates a plausible recreation of the image for visualization
        image = images[r_idx].cpu().numpy().transpose(1, 2, 0)
        image = apply_image_denorm(image)

        rows.append(np.concatenate([image, pred_mask, gt_mask], axis=0))

        ce_losses += ce_loss.item()

    if len(rows) > 10:
        # randomly sample 10 images
        random.shuffle(rows)
        rows = rows[:10]

    mean_iou = iou_metric.compute()
    mean_dice = dice_metric.compute()
    mean_loss = (ce_losses / len(dataloader.dataset)) + (1 - mean_dice)

    img = Image.fromarray(np.concatenate(rows, axis=1).astype(np.uint8))

    return mean_iou.item(), mean_dice.item(), mean_loss, img


def train_simple(
    model_name,
    train_loader,
    val_loader,
    label_mapping,
    test_name,
    experiment_name="stemcv_experiments",
    optimizer_name=Optimizers.AdamWScheduleFree,
    num_epochs=20,
    device="cuda",
    lr=0.05,
):

    test_dir = f"{experiment_name}/{test_name}"
    if os.path.exists(test_dir):
        raise ValueError(
            f"Test with name {test_name} already exists. Please choose a different name."
        )
    os.makedirs(test_dir, exist_ok=True)
    weights_dir = f"{test_dir}/weights"
    os.makedirs(weights_dir, exist_ok=True)
    metrics_dir = f"{test_dir}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    print("Train dataset size:", len(train_loader.dataset))
    print("Val dataset size:", len(val_loader.dataset))

    experiment = get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else create_experiment(
            experiment_name, artifact_location=f"s3://{BUCKET_NAME}"
        )
    )
    set_experiment(experiment_id=experiment_id)

    num_classes, label_mapping = (
        train_loader.num_classes,
        train_loader.label_mapping,
    )

    with mlflow.start_run(run_name=test_name):
        mlflow.log_params(
            {
                "model_name": model_name.value,
                "num_classes": num_classes,
                "optimizer_name": optimizer_name.value,
                "num_epochs": num_epochs,
                "device": device,
                "lr": lr,
            }
        )

        img = display_augs(train_loader)
        img.save(f"{test_dir}/augmentations.jpg")

        mlflow.log_artifact(f"{test_dir}/augmentations.jpg")

        model = build_model_from_name(model_name, num_classes=num_classes).to(
            device
        )
        # Log number of parameters
        mlflow.log_param(
            "num_params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        optimizer, opt_needs_set = build_optimizer_from_name(
            optimizer_name, model, lr=lr
        )
        mlflow.log_metric("lr", lr, step=0)

        if opt_needs_set:
            optimizer.train()
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader.dataset)
                // train_loader.batch_size,
            )
            scheduler=None

        best_val = 0.0
        for epoch in range(1, num_epochs + 1):

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                opt_needs_set,
                device,
                epoch,
                num_epochs,
                scheduler,
            )

            # Needed as specified in
            # https://github.com/facebookresearch/schedule_free?tab=readme-ov-file#caveats
            if opt_needs_set:
                optimizer.eval()
                with torch.no_grad():
                    for images, _ in itertools.islice(train_loader, 50):
                        images = images.to(device, non_blocking=True)
                        model(images)

            val_iou, val_dice, mean_val_loss, img = evaluate(
                model,
                optimizer,
                val_loader,
                label_mapping,
                device,
                opt_needs_set,
                num_classes=num_classes,
            )

            iou_r = round(val_iou, 3)
            dice_r = round(val_dice, 3)
            img.save(
                f"{metrics_dir}/epoch-{epoch}_iou-{iou_r}_dice-{dice_r}_val_output.jpg"
            )

            mlflow.log_metric("val_iou", val_iou, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)
            mlflow.log_metric("val_loss", mean_val_loss, step=epoch)

            log_lr = optimizer.param_groups[0].get(
                "scheduled_lr", optimizer.param_groups[0].get("lr")
            )
            mlflow.log_metric("lr", log_lr, step=epoch)

            mlflow.log_artifact(
                f"{metrics_dir}/epoch-{epoch}_iou-{iou_r}_dice-{dice_r}_val_output.jpg"
            )

            print(f"\nEpoch [{epoch}/{num_epochs}]")
            print(
                f"Train Loss: {train_loss:.4f} | Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}\n"
            )

            if opt_needs_set:
                optimizer.eval()

            if val_dice > best_val:
                best_val = val_dice

                # empty weights dir
                for f in os.listdir(weights_dir):
                    if f.startswith("best_"):
                        os.remove(f"{weights_dir}/{f}")
                torch.save(
                    model.state_dict(),
                    f"{weights_dir}/best_{model_name.value}_{dice_r}.pth",
                )
                print(f"Model saved with Val Dice: {val_dice:.4f}")

            torch.save(
                model.state_dict(),
                f"{weights_dir}/last_{model_name.value}_{dice_r}.pth",
            )
