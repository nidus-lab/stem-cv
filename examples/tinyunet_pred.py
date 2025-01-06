import os

import torch
from PIL import Image

from stem_cv.models.tiny_bi_unet import infer as infer_bi_unet
from stem_cv.models.tiny_unet import infer

label_mapping = {
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
}


def get_model_path(name):
    weight_dir = f"./stemcv_experiments/{name}/weights"
    for file in os.listdir(weight_dir):
        if "best_tiny_unet" in file:
            model_path = os.path.join(weight_dir, file)
            break

    return model_path


# Example usage
if __name__ == "__main__":
    TEST_NAME = "test-baseline2"
    MULTI = True

    image_path = "./shared/study-hip-3dus-chop/huggingface/images/val/PHI054_2_L_69.jpg"  # Replace with the path to your test image
    # Get model with :best_tiny_unet: in name

    if MULTI:
        model_paths = []
        for label in ["label1", "label2", "label3"]:
            model_path = get_model_path(f"{TEST_NAME}-{label}")
            model_paths.append(model_path)

        output_image = infer_bi_unet(
            image_path,
            model_paths,
            label_mapping,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    else:

        weight_dir = f"./stemcv_experiments/{TEST_NAME}/weights"
        for file in os.listdir(weight_dir):
            if "best_tiny_unet" in file:
                model_path = os.path.join(weight_dir, file)
                break

        # Perform inference
        output_image = infer(
            image_path,
            model_path,
            label_mapping,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    print(image_path)

    output_image.save("predicted_mask.png")
    print("Inference completed. Segmentation mask saved to predicted_mask.png")
