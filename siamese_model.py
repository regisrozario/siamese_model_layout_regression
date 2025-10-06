import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super().__init__()
        # ResNet18 backbone up to (and including) global avg pool
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # -> (B,512,1,1)

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        # Head produces a single logit (no Sigmoid here)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1)
        )

        # Back-compat alias so checkpoints with "fc.*" also load
        self.fc = self.head

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return 512-D embedding after GAP."""
        f = self.feature_extractor(x)          # (B,512,1,1)
        return torch.flatten(f, 1)             # (B,512)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_once(input1)         # (B,512)
        f2 = self.forward_once(input2)         # (B,512)
        diff = torch.abs(f1 - f2)              # (B,512)
        logit = self.head(diff)                # (B,1)
        return logit                            # logits; apply sigmoid in predict.py
