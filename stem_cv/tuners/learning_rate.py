import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchmetrics.classification import Dice

from stem_cv.config import (
    Optimizers,
    build_model_from_name,
    build_optimizer_from_name,
)


def find_lr(
    model_name,
    dataloader,
    device,
    optimizer_name=Optimizers.AdamW,
    init_value=1e-8,
    final_value=1,
    beta=0.98,
    stop_div=True,
):
    """
    Implements the "LR range test" approach.
    """

    num_classes = dataloader.num_classes

    model = build_model_from_name(model_name, num_classes=num_classes).to(
        device
    )

    optimizer, opt_needs_set = build_optimizer_from_name(
        optimizer_name, model, lr=init_value
    )

    if opt_needs_set:
        optimizer.train()

    num = len(dataloader) - 1  # total number of batches
    mult = (final_value / init_value) ** (1 / num)

    model.train()

    dice_metric = Dice(num_classes=num_classes, ignore_index=0).to(device)
    avg_loss = 0.0
    best_loss = 0.0
    batch_num = 0

    losses = []
    lrs = []
    lr = init_value

    for i, (images, masks) in enumerate(dataloader):
        batch_num += 1

        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        ce_loss = nn.CrossEntropyLoss()(outputs, masks)
        preds = outputs.argmax(dim=1)
        dsc_loss = 1 - dice_metric(preds, masks)
        dice_metric.reset()

        loss = ce_loss + dsc_loss

        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        if batch_num == 1:
            best_loss = smoothed_loss

        losses.append(smoothed_loss)
        lrs.append(lr)

        # Backward
        loss.backward()
        optimizer.step()

        lr *= mult
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if stop_div and (smoothed_loss > 4 * best_loss or torch.isnan(loss)):
            print("Stopping early, loss has diverged.")
            break

        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

    print(
        "**NOTE: "
        "Look for largest downwards slope in the plot as best learning rate. **"
    )

    # cut off values above losses[0]
    max_loss = losses[0]
    lrs, losses = zip(
        *[(lr, loss) for lr, loss in zip(lrs, losses) if loss <= max_loss]
    )

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(lrs, losses)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Loss")
    ax.set_title("LR Finder")

    # Return the figure object
    return fig
