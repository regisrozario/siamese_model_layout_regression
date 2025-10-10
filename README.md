# Visual Regression Testing with Siamese Neural Networks

A deep learning system for automated visual regression detection in web applications using Siamese Neural Networks with ResNet18 backbone.

## 🎯 Overview

This project implements a binary classification Siamese Neural Network that compares pairs of UI screenshots to detect visual layout regressions. The model learns to distinguish between consistent layouts (baseline vs baseline) and regressed layouts (baseline vs buggy version).

**Key Capability**: Given two screenshots, the model outputs a probability indicating whether they represent the same layout (0) or a regression has occurred (1)

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training dataset
python data_generator.py --samples 50

# 3. Create training pairs CSV
python build_csv.py

# 4. Train the model
python train.py

# 5. Run predictions
python predict.py
```

## 📋 Requirements

```
Python 3.8+
PyTorch 2.1.2
torchvision 0.16.2
OpenCV
Pillow
pandas
numpy
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Architecture

### Siamese Network Design

The model uses a **twin architecture** with shared weights to process two input images:

```
Image 1 (224×224×3)              Image 2 (224×224×3)
        ↓                                ↓
    ResNet18 Encoder              ResNet18 Encoder
    (shared weights)              (shared weights)
        ↓                                ↓
   Embedding (512-D)              Embedding (512-D)
        ↓                                ↓
          LayerNorm              LayerNorm
                ↘                      ↙
                  Fusion Layer (AbsDiff)
                         ↓
                    FC: 512 → 256
                         ↓
                     ReLU + Dropout
                         ↓
                    FC: 256 → 1
                         ↓
                  Logit → Sigmoid
                         ↓
                 Probability [0, 1]
```

### Model Components

**1. Feature Extractor**
- **Backbone**: ResNet18 pretrained on ImageNet
- **Output**: 512-dimensional feature embeddings
- **Grayscale Processing**: Converts RGB to grayscale before feature extraction
- **Normalization**: LayerNorm stabilizes embedding scale

**2. Fusion Strategy**
- **Absolute Difference** (default): `|embed1 - embed2|`
  - Symmetric comparison
  - Focuses on dissimilarity magnitude
- **Concatenation** (optional): `[embed1, embed2]`
  - Asymmetric patterns
  - Learns directional changes

**3. Classification Head**
```python
nn.Sequential(
    nn.Linear(512, 256),      # Dimensionality reduction
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.2),        # Regularization
    nn.Linear(256, 1)         # Binary output
)
```

**4. Loss Function**
- **BCEWithLogitsLoss**: Binary Cross Entropy with built-in sigmoid
- Combines sigmoid activation and BCE in single operation
- More numerically stable than separate operations

### Model Configuration

```python
from siamese_model import SiameseNetwork

model = SiameseNetwork(
    freeze_backbone=True,      # Freeze pretrained ResNet18
    out_dim=1,                 # Binary classification
    fusion="absdiff",          # Symmetric similarity measure
    use_layernorm=True,        # Normalize embeddings
    proj_dim=None,             # Optional projection dimension
    dropout=0.2                # Dropout rate
)
```

**Architecture Variants**:
- **Frozen backbone**: Fast training, prevents overfitting on small datasets
- **Trainable backbone**: Better performance with large datasets (>1000 pairs)
- **Concat fusion**: Use when order matters (e.g., before/after)
- **AbsDiff fusion**: Use when only magnitude of change matters

## 📊 Dataset Format

### Training Data Structure

```
dataset/
├── baseline/           # Clean/reference screenshots
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
├── regressed/         # Screenshots with layout bugs
│   ├── image_001.png  # Same filename = matched pair
│   ├── image_002.png
│   └── ...
└── pairs.csv          # Training manifest
```

### CSV Format

```csv
baseline,candidate,label
dataset/baseline/img1.png,dataset/regressed/img1.png,1
dataset/baseline/img2.png,dataset/baseline/img2.png,0
```

- **Label 0**: Same/consistent layout (baseline vs baseline)
- **Label 1**: Different/regression (baseline vs regressed)

### Data Requirements

| Metric | Minimum | Recommended |
|--------|---------|-------------|
| Total pairs | 50 | 200+ |
| Positive examples (label=1) | 20 | 80+ |
| Negative examples (label=0) | 20 | 80+ |
| Validation samples | 10 | 20+ |

**Balance ratio**: Keep negative:positive between 1:1 and 2:1

## 🎓 Training the Model

### Training Script

```bash
python train.py
```

### Hyperparameters

```python
BATCH_SIZE = 16                  # Batch size
LEARNING_RATE = 1e-4             # Adam optimizer learning rate
NUM_EPOCHS = 25                  # Maximum epochs
EARLY_STOP_PATIENCE = 5          # Stop if no improvement
TRAIN_SPLIT = 0.8                # 80/20 train/validation split
FREEZE_BACKBONE = True           # Freeze ResNet18 weights
```

### Training Pipeline

**1. Data Loading**
```python
# Training augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # RGB grayscale
    transforms.RandomHorizontalFlip(p=0.3),        # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],    # ImageNet stats
                        [0.229, 0.224, 0.225]),
])

# Validation (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
])
```

**2. Model Initialization**
```python
model = SiameseNetwork(freeze_backbone=True).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
```

**3. Training Loop**
- Forward pass through both images
- Compute BCE loss
- Backpropagation
- Optimizer step
- Learning rate scheduling

**4. Validation**
- Compute validation loss and accuracy
- Calculate optimal threshold using F1 score
- Early stopping based on validation loss

**5. Checkpointing**
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'best_threshold': 0.4918,      # Optimal classification threshold
    'best_f1': 0.923,               # F1 score at optimal threshold
    'val_accuracy': 94.2,           # Validation accuracy
    'epoch': 15
}
```

### Training Output

```
Epoch 01/25 | Train Loss: 0.3421 | Val Loss: 0.2134 | Val Acc: 91.23% | LR: 1.00e-04
   ✅ Best model saved! (Optimal threshold: 0.492, F1: 0.889)

Epoch 02/25 | Train Loss: 0.2156 | Val Loss: 0.1823 | Val Acc: 93.45% | LR: 1.00e-04
   ✅ Best model saved! (Optimal threshold: 0.487, F1: 0.912)

...

🛑 Early stopping triggered after 15 epochs

✅ Training completed!
📁 Best model saved to: siamese_model_best.pth
📊 Final validation accuracy: 94.20%
🎯 Recommended threshold: 0.4918
```

### Model Files

- **`siamese_model_best.pth`**: Complete checkpoint with metadata
  - Model weights
  - Optimal threshold
  - Training history
  - Optimizer/scheduler state

- **`siamese_model.pth`**: State dict only (backward compatibility)

## 🔮 Making Predictions

### Using predict.py

```python
import torch
from predict import load_model_and_threshold, predict_regression
from torchvision import transforms

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "siamese_model_best.pth"

# Load model with optimal threshold
model, threshold = load_model_and_threshold(MODEL_PATH, device)

# Define transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Predict
result = predict_regression(
    baseline_path="dataset/test/baseline.png",
    candidate_path="dataset/test/candidate.png",
    model=model,
    transform=transform,
    device=device,
    threshold=threshold,
    show_viz=True,              # Display visual diff
    confidence_margin=0.1        # Confidence band
)

# Result dictionary
print(result)
# {
#     'regression_prob': 0.8234,
#     'similarity': 0.1766,
#     'is_regression': True,
#     'threshold': 0.4918,
#     'confidence': 'HIGH',
#     'distance_from_threshold': 0.3316,
#     'diff_path': 'dataset/test_images/diff_output.png'
# }
```

### Prediction Output

```
============================================================
🧠 Model Prediction Results
============================================================
Regression Probability: 0.8234
Similarity Score:       0.1766
Decision Threshold:     0.4918
Confidence Margin:      ±0.1000
Distance from Threshold: 0.3316
Confidence Level:       HIGH
============================================================
❌ LAYOUT REGRESSION DETECTED!
   The images are significantly different (prob: 82.34%)
   💾 Visual diff saved to: dataset/test_images/diff_output.png
   📦 Found 3 difference region(s)
============================================================
```

### Confidence Levels

**HIGH Confidence**:
- Probability > threshold + margin (e.g., > 0.59)
- Probability < threshold - margin (e.g., < 0.39)
- Clear decision, reliable prediction

**LOW Confidence**:
- Probability within ±margin of threshold (0.39 to 0.59)
- Borderline case, manual review recommended
- Adjust threshold or retrain with more data

### Visual Diff Output

The system generates annotated images with:
- Red bounding boxes around detected differences
- Morphological filtering to reduce noise
- Saved to `diff_output.png`

## ⚙️ Model Inference

### Batch Prediction

```python
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class PairDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs  # List of (baseline_path, candidate_path)
        self.transform = transform
    
    def __getitem__(self, idx):
        baseline_path, candidate_path = self.pairs[idx]
        img1 = Image.open(baseline_path).convert('RGB')
        img2 = Image.open(candidate_path).convert('RGB')
        return self.transform(img1), self.transform(img2)
    
    def __len__(self):
        return len(self.pairs)

# Create dataset
pairs = [
    ("baseline1.png", "candidate1.png"),
    ("baseline2.png", "candidate2.png"),
    # ...
]
dataset = PairDataset(pairs, transform)
loader = DataLoader(dataset, batch_size=32)

# Batch inference
model.eval()
predictions = []
with torch.inference_mode():
    for img1, img2 in loader:
        img1, img2 = img1.to(device), img2.to(device)
        logits = model(img1, img2)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        predictions.extend(probs)

# Apply threshold
is_regression = [p > threshold for p in predictions]
```

### Embedding Extraction

```python
# Extract embeddings for similarity search
model.eval()
with torch.inference_mode():
    img_tensor = transform(img).unsqueeze(0).to(device)
    embedding = model.embed(img_tensor)  # Returns 512-D vector
    
# Use for nearest neighbor search, clustering, etc.
```

## 🎯 Use Cases

1. **CI/CD Pipeline Integration**
   - Automated visual regression testing
   - Block deployments on regressions

2. **A/B Test Validation**
   - Compare design variations
   - Ensure consistency across experiments

3. **Cross-Browser Testing**
   - Detect rendering differences
   - Validate browser compatibility

4. **Responsive Design QA**
   - Compare layouts across viewports
   - Verify mobile responsiveness

## 🔧 Advanced Configuration

### Custom Fusion Strategy

```python
model = SiameseNetwork(
    fusion="concat",           # Use concatenation instead of absdiff
    proj_dim=512,             # Project 1024→512 before head
)
```

### Trainable Backbone

```python
model = SiameseNetwork(
    freeze_backbone=False,     # Fine-tune ResNet18
)
# Requires larger dataset and more epochs
```

### Custom Classification Head

```python
class CustomSiamese(SiameseNetwork):
    def __init__(self):
        super().__init__()
        # Replace head with custom architecture
        self.head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
```

## 📈 Performance Tuning

### Increase Training Data
```bash
# Generate more diverse screenshots
python data_generator.py --samples 100 --urls <url1> <url2> <url3>
```

### Adjust Threshold
```python
# Test different thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for thresh in thresholds:
    result = predict_regression(..., threshold=thresh)
    # Evaluate on validation set
```

### Learning Rate Tuning
```python
# In train.py
LEARNING_RATE = 5e-5  # Lower for fine-tuning
# or
LEARNING_RATE = 1e-3  # Higher for faster convergence
```

### Data Augmentation
```python
# Add more augmentation for robustness
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

---

**Architecture**: Siamese Neural Network | **Backbone**: ResNet18 | **Framework**: PyTorch
```

**Configure paths** in `predict.py`:
```python
BASELINE_PATH = "dataset/test_images/counter_app_clean.png"
CANDIDATE_PATH = "dataset/test_images/counter_app_text_overlap.png"
```

### Prediction Output

```
============================================================
🧠 Model Prediction Results
============================================================
Regression Probability: 0.8234
Similarity Score:       0.1766
Decision Threshold:     0.4918
Confidence Margin:      ±0.1000
Distance from Threshold: 0.3316
Confidence Level:       HIGH
============================================================
❌ LAYOUT REGRESSION DETECTED!
   The images are significantly different (prob: 82.34%)
   💾 Visual diff saved to: dataset/test_images/diff_output.png
   📦 Found 3 difference region(s)
============================================================
```

## 🏗️ Project Structure

```
visual-regression-testing/
├── dataset/
│   ├── baseline/           # Clean screenshots
│   ├── regressed/          # Screenshots with bugs
│   ├── test_images/        # Test images for prediction
│   └── pairs.csv           # Training pairs CSV
├── data_generator.py       # Generate training data
├── build_csv.py            # Create training CSV
├── train.py                # Train the model
├── predict.py              # Make predictions
├── main.py                 # End-to-end prediction pipeline
├── siamese_model.py        # Neural network architecture
├── generate_livescreenshot.py  # Capture live screenshots
├── ModelVerification.py    # Dataset diagnostics
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🧠 Model Architecture

### Siamese Network

The model uses a **Siamese architecture** with shared weights:

```
Input Images (224×224×3)
    ↓
ResNet18 Backbone (pretrained on ImageNet)
    ↓
Feature Embeddings (512-D)
    ↓
Layer Normalization
    ↓
Fusion Layer (AbsDiff or Concat)
    ↓
Fully Connected Head (512→256→1)
    ↓
Output Logit (sigmoid → probability)
```

**Key Components**:
- **Backbone**: ResNet18 pretrained on ImageNet
- **Fusion Strategy**: Absolute difference (symmetric) or concatenation
- **Output**: Single logit → binary classification via BCEWithLogitsLoss
- **Normalization**: Optional LayerNorm for embedding stability

### Model Configuration

```python
model = SiameseNetwork(
    freeze_backbone=True,      # Freeze ResNet18 weights
    out_dim=1,                 # Binary classification
    fusion="absdiff",          # Symmetric fusion
    use_layernorm=True,        # Normalize embeddings
    dropout=0.2                # Regularization
)
```

## ⚙️ Configuration

### URLs to Test

Edit `data_generator.py`:
```python
URLS_TO_TEST = [
    "https://www.google.com",
    "https://your-app.com",
    "https://another-page.com",
]
```

### Bug Injection Severity

The system injects 7 types of layout bugs:
1. Hidden elements
2. Shifted elements
3. Broken sizing
4. Overlapping elements
5. Broken text/content
6. Hidden sections
7. Broken alignment

Control severity in `_inject_random_layout_bugs()`:
- `light`: 30% chance per bug type
- `medium`: 60% chance per bug type
- `heavy`: 90% chance per bug type

### Prediction Threshold

The optimal threshold is automatically calculated during training and saved in the model checkpoint. You can override it:

```python
# In predict.py
result = predict_regression(
    baseline_path,
    candidate_path,
    model,
    transform,
    device,
    threshold=0.5,           # Custom threshold
    confidence_margin=0.1    # Confidence band
)
```

## 📊 Dataset Recommendations

For reliable training:

- **Minimum**: 50 total pairs (25 positive + 25 negative)
- **Good**: 100-200 pairs
- **Optimal**: 300+ pairs from diverse URLs

**Class Balance**:
- Target ratio: 1:1 to 2:1 (negative:positive)
- Handled automatically by `build_csv.py`

**Validation Set**:
- Minimum: 10 samples (system warns if less)
- Recommended: 20+ samples

## 🔍 Diagnostics

Check dataset health:

```bash
python ModelVerification.py
```

Output:
```
============================================================
DATASET DIAGNOSIS
============================================================

Total pairs: 150

Class distribution:
0    90
1    60

Label balance:
0    0.6
1    0.4

Expected split (80/20):
  Training: 120 samples
  Validation: 30 samples

Duplicate rows: 0
Same-image pairs: 90 (60.0%)
```

## 🎨 Visual Diff Generation

The system automatically generates visual diffs with:
- Red bounding boxes around detected changes
- Morphological operations to reduce noise
- Contour detection for difference regions
- Saved to `dataset/test_images/diff_output.png`

## 📈 Training Tips

1. **Start Small**: Use `--quick-test` to verify pipeline
2. **Increase Gradually**: Add more URLs and samples
3. **Monitor Overfitting**: Watch validation loss vs train loss
4. **Adjust Threshold**: Use F1 score for optimal threshold
5. **Balance Data**: Keep positive/negative ratio under 3:1
6. **Freeze Backbone**: Speeds up training, prevents overfitting

## 🐛 Troubleshooting

### "No baseline images found"
```bash
# Run data generator first
python data_generator.py --samples 20
```

### "Validation set very small"
```bash
# Generate more data
python data_generator.py --samples 50
```

### "Low confidence predictions"
- Increase training data
- Adjust confidence margin
- Review threshold selection

### ChromeDriver Issues
- Download matching ChromeDriver version
- Add to PATH or specify location
- Check Chrome browser version compatibility

### CUDA Out of Memory
```python
# Reduce batch size in train.py
BATCH_SIZE = 8  # or 4
```

## 📝 Example Workflow

```bash
# 1. Generate diverse dataset
python data_generator.py --samples 50 --urls \
    https://google.com \
    https://github.com \
    https://stackoverflow.com

# 2. Build training pairs
python build_csv.py --preview

# 3. Check dataset health
python ModelVerification.py

# 4. Train model
python train.py

# 5. Test predictions
python main.py --type good
python main.py --type bad

# 6. Use in production
python predict.py  # With custom baseline/candidate paths
```

## 🎯 Use Cases

- **CI/CD Integration**: Automated visual regression testing
- **A/B Testing**: Compare design variations
- **Cross-Browser Testing**: Detect rendering differences
- **Responsive Design**: Verify layouts across viewports
- **Component Testing**: Validate UI component changes

## 🔮 Future Enhancements

- [ ] Multi-class classification (severity levels)
- [ ] Attention maps for explainability
- [ ] Real-time streaming predictions
- [ ] Docker containerization
- [ ] REST API endpoint
- [ ] Integration with test frameworks (Selenium, Playwright)
- [ ] Pixel-level segmentation masks
- [ ] Support for mobile screenshots

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]

---

**Built with PyTorch** | **Powered by ResNet18** | **Siamese Networks for Visual Regression**
