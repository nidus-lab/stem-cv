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
from stem_cv.config import (
    Optimizers,
    build_model_from_name,
    build_optimizer_from_name,
)
from stem_cv.datasets import apply_image_denorm, display_augs
from stem_cv.datasets.process import (
    keep_largest_components,
    postprocess_output,
)
from torchmetrics.classification import Dice, MulticlassJaccardIndex

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
    running_ce_loss = 0.0
    running_dsc_loss = 0.0

    dice_metric = Dice(num_classes=dataloader.num_classes, ignore_index=0).to(
        device
    )

    # Get total number of images
    total_images = len(dataloader.dataset)
    batch_size = dataloader.batch_size

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
        running_ce_loss += ce_loss.item() * images.size(0)
        running_dsc_loss += dsc_loss.item() * images.size(0)

        if (i + 1) % (int(total_images / 10 / batch_size) + 1) == 0 or i == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}"
            )

            mlflow.log_metric(
                "train_loss", running_loss / ((i + 1) * batch_size), step=i
            )
            mlflow.log_metric(
                "train_ce_loss",
                running_ce_loss / ((i + 1) * batch_size),
                step=i,
            )
            mlflow.log_metric(
                "train_dsc_loss",
                running_dsc_loss / ((i + 1) * batch_size),
                step=i,
            )

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
    largest_component_only=True,
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

        if largest_component_only:
            pred_class = keep_largest_components(outputs, num_classes)
        else:
            pred_class = outputs.argmax(dim=1)

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

        # make the image RGB
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

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


def convert_keys_to_strings(input_dict):
    return {str(k): v for k, v in input_dict.items()}


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
    largest_component_only=True,
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
                "batch_size": train_loader.batch_size,
            }
        )

        label_mapping_log = convert_keys_to_strings(label_mapping)
        mlflow.log_dict(label_mapping_log, "label_mapping.json")
        mlflow.log_dict(train_loader.transforms_dict, "train_transforms.json")

        img = display_augs(train_loader)
        img.save(f"{test_dir}/augmentations.jpg")

        mlflow.log_artifact(f"{test_dir}/augmentations.jpg")

        in_channels = 3 if train_loader.dataset[0][0].shape[0] == 3 else 1
        model = build_model_from_name(
            model_name, num_classes=num_classes, in_channels=in_channels
        ).to(device)
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs
            )

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
                model.train()
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
                largest_component_only=largest_component_only,
            )

            iou_r = round(val_iou, 3)
            dice_r = round(val_dice, 3)
            # add leading 0's to epoch
            epoch_name = str(epoch).zfill(len(str(num_epochs)))
            img.save(
                f"{metrics_dir}/epoch-{epoch_name}_iou-{iou_r}_dice-{dice_r}_val_output.jpg"
            )

            mlflow.log_metric("val_iou", val_iou, step=epoch)
            mlflow.log_metric("val_dice", val_dice, step=epoch)
            mlflow.log_metric("val_loss", mean_val_loss, step=epoch)

            log_lr = optimizer.param_groups[0].get(
                "scheduled_lr", optimizer.param_groups[0].get("lr")
            )
            mlflow.log_metric("lr", log_lr, step=epoch)

            mlflow.log_artifact(
                f"{metrics_dir}/epoch-{epoch_name}_iou-{iou_r}_dice-{dice_r}_val_output.jpg"
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
