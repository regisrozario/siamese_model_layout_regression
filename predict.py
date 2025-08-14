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
DIFF_OUTPUT_PATH = "dataset/test_images/diff_login.png"
SIMILARITY_THRESHOLD = 0.5
# -------------------------------------

# Load model
model = SiameseNetwork()
model.load_state_dict(torch.load("siamese_model.pth", map_location=torch.device("cpu")))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load and transform images
img1 = Image.open(BASELINE_PATH).convert("RGB")
img2 = Image.open(CANDIDATE_PATH).convert("RGB")
t1 = transform(img1).unsqueeze(0)
t2 = transform(img2).unsqueeze(0)

# Predict similarity
with torch.no_grad():
    similarity = model(t1, t2).item()

print(f"\n🧠 Similarity Score: {similarity:.4f}")

# Decide and output message
if similarity >= SIMILARITY_THRESHOLD:
    print("❌ Layout Regression Detected!")


    # Generate diff image
    def save_visual_diff(img1_path, img2_path, output_path):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imwrite(output_path, img2)
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
