from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class SiameseNetwork(nn.Module):
    """
    Why: Make the original clean model usable for both classification (1 logit) and regression (N outputs),
    optionally improve stability via embedding normalization and flexible fusion strategies.
    """

    def __init__(
        self,
        freeze_backbone: bool = False,
        out_dim: int = 1,                 # 1 for binary classification; >1 for regression/multi-label
        fusion: str = "absdiff",          # "absdiff" or "concat"
        use_layernorm: bool = True,       # normalizes 512-D embeddings before fusion
        proj_dim: int | None = None,      # if set with fusion="concat", project 1024->proj_dim before head
        dropout: float = 0.2,
    ):
        super().__init__()
        if fusion not in {"absdiff", "concat"}:
            raise ValueError(f'Unsupported fusion="{fusion}". Use "absdiff" or "concat".')

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # -> (B,512,1,1)

        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        self.use_layernorm = use_layernorm
        self.ln = nn.LayerNorm(512) if use_layernorm else nn.Identity()
        self.fusion = fusion

        # Determine head input dimension
        fused_dim = 512 if fusion == "absdiff" else 512 * 2
        if proj_dim is not None:
            self.proj = nn.Linear(fused_dim, proj_dim, bias=True)
            head_in = proj_dim
        else:
            self.proj = nn.Identity()
            head_in = fused_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, out_dim),
        )

        # Back-compat alias so checkpoints with "fc.*" also load
        self.fc = self.head

    @property
    def feature_dim(self) -> int:
        return 512

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature_extractor(x)   # (B,512,1,1)
        z = torch.flatten(f, 1)         # (B,512)
        if self.use_layernorm:
            z = self.ln(z)              # Why: stabilizes scale across batches/pairs.
        return z

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)

    def fuse(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        if self.fusion == "absdiff":
            return torch.abs(f1 - f2)   # symmetric similarity
        else:
            return torch.cat([f1, f2], dim=1)  # allows asymmetric patterns

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        f1 = self.forward_once(input1)          # (B,512)
        f2 = self.forward_once(input2)          # (B,512)
        fused = self.fuse(f1, f2)               # (B,512) or (B,1024)
        fused = self.proj(fused)                # (B,proj_dim or fused_dim)
        out = self.head(fused)                  # (B,out_dim)
        return out