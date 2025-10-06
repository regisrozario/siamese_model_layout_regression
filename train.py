import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from siamese_model import SiameseNetwork

class LayoutDataset(Dataset):
    def __init__(self, csv_file, base_dir, transform=None):
        self.pairs = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        img1 = Image.open(row['baseline']).convert('RGB')
        img2 = Image.open(row['candidate']).convert('RGB')
        label = torch.tensor(float(row['label']), dtype=torch.float32)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = LayoutDataset('dataset/pairs.csv', base_dir='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SiameseNetwork(freeze_backbone=True).to(device)
crit  = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for img1, img2, label in dataloader:
        optimizer.zero_grad()
        outputs = model(img1, img2).squeeze()
        loss = crit(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "siamese_model.pth")
