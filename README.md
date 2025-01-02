# STEM-CV

A Computer Vision AI Library designed to be used by anyone in STEM.

Comes with default models, and tools to create your own models.

## Dataset Expected Format

The dataset should be in the following format:

```
dataset
│
└───images
│   │
│   └───train
│   │   │   image1.jpg
│   └───val
│       │   image2.jpg
│
└───masks
    │
    └───train
    │   │   image1.png
    └───val
        │   image2.png
```

Where .jpg files are the images and .png files are the masks, as RGB images.

**NOTE: Black (0,0,0) is reserved for background in masks!**

## Credits

- TinyU-Net: https://github.com/ChenJunren-Lab/TinyU-Net (MIT License)