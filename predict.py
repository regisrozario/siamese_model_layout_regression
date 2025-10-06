import os
import cv2
import torch
from PIL import Image
from siamese_model import SiameseNetwork
from torchvision import transforms
import matplotlib.pyplot as plt

# --------- Configurable paths ---------
BASELINE_PATH = "dataset/test_images/counter_app_clean.png"
CANDIDATE_PATH = "dataset/test_images/counter_app_text_overlap.png"
DIFF_OUTPUT_PATH = "dataset/test_images/count_diff.png"

# Tune this on your validation set (higher = stricter "similar")
SIMILARITY_THRESHOLD = 0.60
# -------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = SiameseNetwork().to(device)

# Load state dict (supports both "fc.*" and "head.*")
state = torch.load("siamese_model.pth", map_location="cpu")
if isinstance(state, dict) and "model" in state:
    state = state["model"]  # full checkpoint -> model weights

# If checkpoint uses 'fc.' names, remap to 'head.' (or vice versa)
if any(k.startswith("fc.") for k in state.keys()):
    state = { (k.replace("fc.", "head.") if k.startswith("fc.") else k): v for k, v in state.items() }

model.load_state_dict(state, strict=False)
model.eval()

# Transform: include ImageNet normalization for ResNet18
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def load_tensor(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

# Load and transform images
t1 = load_tensor(BASELINE_PATH)
t2 = load_tensor(CANDIDATE_PATH)

# Predict similarity (apply sigmoid here; model returns logits)
with torch.inference_mode():
    logit = model(t1, t2).view(-1)[0]      # scalar logit
    similarity = torch.sigmoid(logit).item()

print(f"\n🧠 Similarity Score: {similarity:.4f}")

# Decide and output message
# If similarity is LOW, we flag a regression.
is_regression = (similarity < SIMILARITY_THRESHOLD)

if is_regression:
    print("❌ Layout Regression Detected!")

    def save_visual_diff(img1_path: str, img2_path: str, output_path: str) -> str:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            raise FileNotFoundError("One of the input images could not be read by OpenCV.")

        # Resize candidate if needed to match baseline
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Simple pixel diff with bounding boxes
        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img2.copy()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # filter tiny boxes (noise)
            if w * h < 50:
                continue
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imwrite(output_path, overlay)
        return output_path

    diff_path = save_visual_diff(BASELINE_PATH, CANDIDATE_PATH, DIFF_OUTPUT_PATH)

    # Display image
    img = cv2.cvtColor(cv2.imread(diff_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title("Visual Diff - Layout Regression Detected")
    plt.axis('off')
    plt.show()
else:
    print("✅ Layout is consistent. No regression detected.")
