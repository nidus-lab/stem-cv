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
    image_path = "./shared/study-hip-3dus-chop/huggingface/images/val/PHI054_2_L_69.jpg"  # Replace with the path to your test image
    model_path = "weights/best_tiny_unet.pth"  # Path to the trained model file

    # Perform inference
    pred_mask = infer(
        image_path,
        model_path,
        label_mapping,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print("huggingface/images/val/PHI054_2_L_69.jpg")

    # Save the output mask
    output_image = Image.fromarray(
        pred_mask
    )  # Scale to 0-255 for visualization
    output_image.save("predicted_mask.png")
    print("Inference completed. Segmentation mask saved to predicted_mask.png")
