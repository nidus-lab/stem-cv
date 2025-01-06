import os

import torch
from PIL import Image

from stem_cv.models.tiny_unet import infer

label_mapping = {
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
}

# Example usage
if __name__ == "__main__":
    TEST_NAME = "test-og-optstrongaugsv2-elasticstrong-dice"

    image_path = "./shared/study-hip-3dus-chop/huggingface/images/val/PHI054_2_L_69.jpg"  # Replace with the path to your test image
    # Get model with :best_tiny_unet: in name
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
