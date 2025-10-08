import os
import cv2
import torch
from PIL import Image
from siamese_model import SiameseNetwork
from torchvision import transforms
import matplotlib.pyplot as plt
import argparse

# --------- Configurable paths ---------
BASELINE_PATH = "dataset/test_images/counter_app_clean.png"
CANDIDATE_PATH = "dataset/test_images/counter_app_text_overlap.png"
DIFF_OUTPUT_PATH = "dataset/test_images/diff_output.png"

# Model checkpoint path
MODEL_PATH = "siamese_model_best.pth"  # Use best model with optimal threshold


# -------------------------------------

def load_model_and_threshold(model_path: str, device: torch.device):
    """Load model and optimal threshold from checkpoint."""
    model = SiameseNetwork().to(device)

    if os.path.exists(model_path):
        print(f"📦 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # Full checkpoint with metadata
                model.load_state_dict(checkpoint['model_state_dict'])
                threshold = checkpoint.get('best_threshold', 0.5)
                f1_score = checkpoint.get('best_f1', None)
                epoch = checkpoint.get('epoch', 'N/A')
                val_acc = checkpoint.get('val_accuracy', None)

                print(f"   ✅ Loaded checkpoint from epoch {epoch}")
                if val_acc:
                    print(f"   📊 Validation accuracy: {val_acc:.2f}%")
                if f1_score:
                    print(f"   🎯 Optimal threshold: {threshold:.4f} (F1: {f1_score:.3f})")
                else:
                    print(f"   🎯 Using threshold: {threshold:.4f}")

                return model, threshold

            elif 'model' in checkpoint:
                # Legacy format
                model.load_state_dict(checkpoint['model'])
                print("   ⚠️  Legacy checkpoint - using default threshold 0.5")
                return model, 0.5

            else:
                # Just state dict
                model.load_state_dict(checkpoint)
                print("   ⚠️  No threshold in checkpoint - using default 0.5")
                return model, 0.5
        else:
            print("   ⚠️  Unknown checkpoint format - using default threshold 0.5")
            return model, 0.5
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")


def load_and_transform_image(path: str, transform, device: torch.device) -> torch.Tensor:
    """Load image and apply transforms."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def save_visual_diff(baseline_path: str, candidate_path: str, output_path: str) -> str:
    """Generate visual diff with red bounding boxes around differences."""
    baseline = cv2.imread(baseline_path)
    candidate = cv2.imread(candidate_path)

    if baseline is None or candidate is None:
        raise FileNotFoundError("One of the input images could not be read by OpenCV.")

    # Resize candidate to match baseline dimensions
    if baseline.shape != candidate.shape:
        candidate = cv2.resize(candidate, (baseline.shape[1], baseline.shape[0]))

    # Calculate pixel-wise difference
    diff = cv2.absdiff(baseline, candidate)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold to create binary mask
    _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on candidate image
    overlay = candidate.copy()
    diff_count = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter tiny boxes (noise)
        if w * h < 100:  # Increased threshold
            continue

        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)
        diff_count += 1

    # Save result
    cv2.imwrite(output_path, overlay)
    print(f"   💾 Visual diff saved to: {output_path}")
    print(f"   📦 Found {diff_count} difference region(s)")

    return output_path


def predict_regression(baseline_path: str, candidate_path: str,
                       model, transform, device: torch.device,
                       threshold: float, show_viz: bool = True,
                       confidence_margin: float = 0.1) -> dict:
    """
    Predict if there's a layout regression between two images.

    Args:
        confidence_margin: Minimum difference from threshold to be confident
                          (default 0.1 means need >10% difference from threshold)

    Returns:
        dict with keys: 'regression_prob', 'is_regression', 'similarity', 'diff_path', 'confidence'
    """
    # Load images
    baseline_tensor = load_and_transform_image(baseline_path, transform, device)
    candidate_tensor = load_and_transform_image(candidate_path, transform, device)

    # Predict
    model.eval()
    with torch.inference_mode():
        logit = model(baseline_tensor, candidate_tensor).view(-1)[0]
        regression_prob = torch.sigmoid(logit).item()

    # Interpret results (label=1 means regression/different)
    similarity = 1.0 - regression_prob  # Convert to similarity score

    # Calculate confidence (how far from threshold)
    distance_from_threshold = abs(regression_prob - threshold)
    is_confident = distance_from_threshold >= confidence_margin

    # Decision logic:
    # 1. If prob > threshold + margin → REGRESSION (confident fail)
    # 2. If prob < threshold - margin → PASS (confident pass)
    # 3. If within margin of threshold → LOW CONFIDENCE (borderline)

    if regression_prob > threshold + confidence_margin:
        # Confident regression detection
        is_regression = True
        confidence_level = 'HIGH'
    elif regression_prob < threshold - confidence_margin:
        # Confident pass
        is_regression = False
        confidence_level = 'HIGH'
    else:
        # Borderline case - within margin of threshold
        is_regression = regression_prob > threshold  # Use threshold as tiebreaker
        confidence_level = 'LOW'

    result = {
        'regression_prob': regression_prob,
        'similarity': similarity,
        'is_regression': is_regression,
        'threshold': threshold,
        'confidence': confidence_level,
        'distance_from_threshold': distance_from_threshold,
        'diff_path': None
    }

    # Print results
    print(f"\n{'=' * 60}")
    print(f"🧠 Model Prediction Results")
    print(f"{'=' * 60}")
    print(f"Regression Probability: {regression_prob:.4f}")
    print(f"Similarity Score:       {similarity:.4f}")
    print(f"Decision Threshold:     {threshold:.4f}")
    print(f"Confidence Margin:      ±{confidence_margin:.4f}")
    print(f"Distance from Threshold: {distance_from_threshold:.4f}")
    print(f"Confidence Level:       {confidence_level}")
    print(f"{'=' * 60}")

    # Handle low confidence cases (borderline)
    if confidence_level == 'LOW':
        print("⚠️  LOW CONFIDENCE PREDICTION")
        print(f"   Probability ({regression_prob:.2%}) is close to threshold ({threshold:.2%})")
        print(f"   Difference: {distance_from_threshold:.2%} (confidence margin: ±{confidence_margin:.2%})")

        if is_regression:
            print(f"   🔸 Borderline case - flagging as REGRESSION")
            print(f"   Consider reviewing manually or adjusting threshold")
        else:
            print(f"   🔸 Borderline case - marking as PASS")
            print(f"   Consider reviewing manually if this seems wrong")
        print(f"{'=' * 60}\n")
        # Don't return early - continue to show results

    # Show final decision
    if is_regression and confidence_level == 'HIGH':
        print("❌ LAYOUT REGRESSION DETECTED!")
        print(f"   The images are significantly different (prob: {regression_prob:.2%})")

        # Generate visual diff
        diff_path = save_visual_diff(baseline_path, candidate_path, DIFF_OUTPUT_PATH)
        result['diff_path'] = diff_path

        # Display image
        if show_viz:
            img = cv2.cvtColor(cv2.imread(diff_path), cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Layout Regression Detected (Prob: {regression_prob:.2%})",
                      fontsize=14, color='red', fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    elif not is_regression and confidence_level == 'HIGH':
        print("✅ LAYOUT IS CONSISTENT")
        print(f"   No regression detected (similarity: {similarity:.2%})")

    elif is_regression and confidence_level == 'LOW':
        # Low confidence regression - show diff but mark as uncertain
        print("⚠️  POSSIBLE REGRESSION (Low Confidence)")

        diff_path = save_visual_diff(baseline_path, candidate_path, DIFF_OUTPUT_PATH)
        result['diff_path'] = diff_path

        if show_viz:
            img = cv2.cvtColor(cv2.imread(diff_path), cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.title(f"Possible Regression - Low Confidence (Prob: {regression_prob:.2%})",
                      fontsize=14, color='orange', fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()

    else:  # not is_regression and confidence_level == 'LOW'
        print("✅ LIKELY CONSISTENT (Low Confidence)")
        print(f"   Probably no regression (similarity: {similarity:.2%})")

    print(f"{'=' * 60}\n")

    return result