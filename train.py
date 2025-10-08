import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
from pathlib import Path

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


def train_model():
    """Main training function."""
    # Hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 25
    EARLY_STOP_PATIENCE = 5
    TRAIN_SPLIT = 0.8
    FREEZE_BACKBONE = True

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # ← ADD THIS LINE
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # ← ADD THIS LINE
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load full dataset
    full_dataset = LayoutDataset('dataset/pairs.csv', base_dir='dataset', transform=None)

    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply different transforms to train and val
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create dataloaders (num_workers=0 for Windows compatibility)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"📊 Dataset split: {train_size} train, {val_size} validation")

    # Initialize model, loss, optimizer
    model = SiameseNetwork(freeze_backbone=FREEZE_BACKBONE).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    print("\n🚀 Starting training...\n")

    for epoch in range(NUM_EPOCHS):
        # ============= Training Phase =============
        model.train()
        train_loss = 0.0

        for img1, img2, label in train_loader:
            # Move to device
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(img1, img2).squeeze()
            loss = criterion(outputs, label)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ============= Validation Phase =============
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                outputs = model(img1, img2).squeeze()
                loss = criterion(outputs, label)
                val_loss += loss.item()

                # Calculate accuracy (threshold = 0.5)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct += (preds == label).sum().item()
                total += label.size(0)

                # Store for threshold tuning
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_accuracy)
        history['learning_rate'].append(current_lr)

        # Print epoch summary
        print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.2e}")

        # ============= Model Checkpointing =============
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Calculate optimal threshold on validation set
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            # Find threshold that maximizes F1 score
            thresholds = np.linspace(0.1, 0.9, 50)
            f1_scores = []
            for thresh in thresholds:
                preds = (all_probs > thresh).astype(float)
                tp = np.sum((preds == 1) & (all_labels == 1))
                fp = np.sum((preds == 1) & (all_labels == 0))
                fn = np.sum((preds == 0) & (all_labels == 1))

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                f1_scores.append(f1)

            best_threshold = thresholds[np.argmax(f1_scores)]
            best_f1 = np.max(f1_scores)

            # Save best model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'history': history
            }

            torch.save(checkpoint, "siamese_model_best.pth")
            torch.save(model.state_dict(), "siamese_model.pth")  # For backward compatibility

            print(f"   ✅ Best model saved! (Optimal threshold: {best_threshold:.3f}, F1: {best_f1:.3f})")
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter} epoch(s)")

        # ============= Early Stopping =============
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch + 1} epochs")
            break

        print()

    print("\n✅ Training completed!")
    print(f"📁 Best model saved to: siamese_model_best.pth")
    print(f"📊 Final validation accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"📉 Best validation loss: {best_val_loss:.4f}")

    # Load and display optimal threshold
    checkpoint = torch.load("siamese_model_best.pth", map_location=device)
    print(f"\n🎯 Recommended threshold for predict.py: {checkpoint['best_threshold']:.4f}")
    print(f"   (This threshold achieved F1 score: {checkpoint['best_f1']:.4f})")


if __name__ == '__main__':
    train_model()