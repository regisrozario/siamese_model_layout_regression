import argparse

import torch

from predict import load_model_and_threshold, predict_regression
from torchvision import transforms
from generate_broken_livescreenshot import take_screenshot

TEST_IMAGES_DIR = "./dataset/test_images"
BASE_IMAGES_DIR = "./dataset/test_images/google_home_clean.png"
BROKEN_IMAGE = "./dataset/test_images/broken.png"
GOOD_IMAGE = "./dataset/test_images/working.png"
MODEL_PATH = "siamese_model_best.pth"
THRESHOLD = 0.4918  # Optimal threshold from training
URL = "https://google.com"


def main():
    ap = argparse.ArgumentParser(description="Predict with Siamese model (classification/regression).")
    ap.add_argument("--type", type=str, required=True, help="Test case type: good or bad")
    args = ap.parse_args()

    if args.type == "good":
        take_screenshot(URL, "good", "working.png")
        candidate_path = GOOD_IMAGE
    else:
        take_screenshot(URL, "bad", "broken.png")
        candidate_path = BROKEN_IMAGE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}\n")

    # Load model and threshold
    model, optimal_threshold = load_model_and_threshold(MODEL_PATH, device)

    # Use custom threshold if provided

    # Define transform (must match training)
    # New (Grayscale)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # Convert to grayscale but keep 3 channels
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Predict
    result = predict_regression(
        BASE_IMAGES_DIR,
        candidate_path,
        model,
        transform,
        device,
        THRESHOLD,
        show_viz=False,
        confidence_margin=0.1
    )
    print("Prediction result:", result)
    return result



if __name__ == "__main__":
    main()
