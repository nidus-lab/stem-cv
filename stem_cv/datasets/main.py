import glob
import os
import random

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .process import postprocess_output


def get_train_transforms(rootdir, label_mapping, patch_size):
    return A.Compose(
        [
            A.GridElasticDeform(
                num_grid_xy=[4, 4],
                magnitude=15,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0,
                rotate_limit=30,
                interpolation=1,  # cv2.INTER_LINEAR
                border_mode=0,  # cv2.BORDER_CONSTANT
            ),
            A.Affine(shear=20, p=0.5),
            A.Affine(scale=(0.6, 1.4), p=0.5),
            A.MultiplicativeNoise(
                multiplier=(0.9, 1.1),
                per_channel=False,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.5,
            ),
            A.Downscale(scale_min=0.5, scale_max=1.0, p=0.25),
            # ========================
            A.Resize(
                patch_size[0], patch_size[1], p=1.0
            ),  # Ensures output matches patch size
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )


DEFAULT_TRANSFORM_VAL = A.Compose(
    [
        A.Resize(256, 256),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)


def mask2label(mask, label_mapping):
    """
    Convert a mask to a label image based on a label mapping.
    """
    label_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    for color, label in label_mapping.items():
        tmask = np.all(mask == color, axis=-1)
        label_mask[tmask] = label
    return label_mask


class SegmentationDataset(Dataset):
    """
    A simple Dataset for segmentation tasks.
    Expects directory structure:
        root_dir/images/*.jpg|.png|...
        root_dir/labels/*.png|.jpg|...
    The naming convention is that the mask has the same base filename
    as the image but with possible extension differences.
    """

    def __init__(
        self,
        root_dir,
        label_mapping,
        transform=None,
        greyscale=False,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.image_folder = root_dir
        self.mask_folder = root_dir.replace("images", "labels")
        self.transform = transform
        self.label_mapping = label_mapping
        self.greyscale = greyscale

        self.image_paths = sorted(
            glob.glob(os.path.join(self.image_folder, "*.*"))
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(
            self.mask_folder, filename.split(".")[0] + ".png"
        )

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("RGB")
        unique_colors = mask_image.getcolors()

        # Basic check for unexpected colors
        for _, color in unique_colors:
            if color == (0, 0, 0):
                continue
            if color not in self.label_mapping:
                raise ValueError(
                    f"Color {color} in mask {mask_path} not found in label mapping."
                )

        # Convert to NumPy
        image = np.array(image, dtype=np.uint8)
        mask_image = np.array(mask_image, dtype=np.uint8)

        if self.greyscale:
            # only take first channel (i.e make greyscale)
            image = image[:, :, 0]

        # Create a 2D mask based on the label mapping
        mask = mask2label(mask_image, self.label_mapping)

        augmented = self.transform(image=image, mask=mask)

        image = augmented["image"]
        mask = augmented["mask"]

        # convert mask to long tensor
        mask = mask.long()

        return image, mask


def get_dataloaders(
    train_root,
    val_root,
    batch_size,
    label_mapping,
    num_workers=4,
    train_transform=None,
    val_transform=None,
    train_pct=1.0,
    val_pct=1.0,
    greyscale=False,
):
    """
    Creates and returns (train_loader, val_loader).
    train_transform, val_transform should be Albumentations
    or None for fallback transforms.
    """
    import copy

    if train_transform is None:
        train_transform = get_train_transforms(
            train_root, label_mapping, (256, 256)
        )
    if val_transform is None:
        val_transform = DEFAULT_TRANSFORM_VAL

    wrapped_transform = A.ReplayCompose(
        copy.deepcopy(train_transform.transforms)
    )

    # Apply the ReplayCompose
    image = np.random.randint(
        0, 256, (512, 512, 3), dtype=np.uint8
    )  # Example image
    augmented = wrapped_transform(image=image)

    train_dataset = SegmentationDataset(
        root_dir=train_root,
        label_mapping=label_mapping,
        transform=train_transform,
        greyscale=greyscale,
    )
    val_dataset = SegmentationDataset(
        root_dir=val_root,
        label_mapping=label_mapping,
        transform=val_transform,
        greyscale=greyscale,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Store some metadata on loaders
    num_classes = len(np.unique(list(label_mapping.values()))) + 1
    train_loader.num_classes = num_classes
    val_loader.num_classes = num_classes
    train_loader.label_mapping = label_mapping
    val_loader.label_mapping = label_mapping
    train_loader.transforms_dict = augmented["replay"]

    # Optionally reduce dataset size
    train_len = int(train_pct * len(train_dataset))
    val_len = int(val_pct * len(val_dataset))
    train_loader.dataset.image_paths = train_loader.dataset.image_paths[
        :train_len
    ]
    val_loader.dataset.image_paths = val_loader.dataset.image_paths[:val_len]

    return train_loader, val_loader


def display_augs(train_loader):
    """
    Display augmented images and masks in a grid with 3 columns and 10 rows.

    :param train_loader: DataLoader with SegmentationDataset

    :return: PIL Image with 3 columns and 10 rows of image-mask pairs
    """
    # Randomly sample 30 image-mask pairs
    sample = random.sample(range(len(train_loader.dataset)), 30)

    rows = []

    for i in range(0, len(sample), 3):  # Process 3 items per row
        row_images = []
        for idx in sample[i : i + 3]:
            image, mask = train_loader.dataset[idx]

            # Convert the image and mask for visualization
            image = image.permute(1, 2, 0).numpy()
            image = apply_image_denorm(image)

            # make the image RGB
            if image.shape[-1] == 1:
                image = np.repeat(image, 3, axis=-1)

            mask = postprocess_output(
                mask, train_loader.label_mapping, needs_argmax=False
            )

            # Concatenate the image and mask side-by-side
            pair = np.concatenate([image, mask], axis=1)
            row_images.append(pair)

        # Concatenate the pairs horizontally to form a row
        if len(row_images) == 3:
            rows.append(np.concatenate(row_images, axis=1))

    # Concatenate all rows vertically to form the grid
    grid = np.concatenate(rows, axis=0)

    # Convert the grid to a PIL Image and return
    img = Image.fromarray(grid.astype(np.uint8))

    return img


def apply_image_denorm(image):
    z_max = image.max()
    z_min = image.min()
    image = (image - z_min) / (z_max - z_min) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image
