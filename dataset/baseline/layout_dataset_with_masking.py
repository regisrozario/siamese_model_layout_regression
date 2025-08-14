import os
import cv2
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image
from safetensors import torch
import torch
from torch.utils.data import Dataset


def mask_text_regions_pil(image: Image.Image) -> Image.Image:
    """
    Mask text regions in a PIL image using Tesseract OCR.
    """
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_image.shape
    boxes = pytesseract.image_to_boxes(cv_image)

    for b in boxes.splitlines():
        b = b.split()
        if len(b) >= 5:
            x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(cv_image, (x1, h - y1), (x2, h - y2), (128, 128, 128), -1)

    masked = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(masked)

class LayoutDatasetWithMasking(Dataset):
    def __init__(self, csv_file, transform=None, mask_text=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.mask_text = mask_text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img1 = Image.open(row['baseline']).convert('RGB')
        img2 = Image.open(row['candidate']).convert('RGB')

        if self.mask_text:
            img1 = mask_text_regions_pil(img1)
            img2 = mask_text_regions_pil(img2)

        label = torch.tensor(float(row['label']), dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label
