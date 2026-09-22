"""
RiskProp model -- same encoder-only shape as the original paper's Fig. 2:
snippet -> f_theta (SlowOnly-R50) -> risk representation z_t -> (sigmoid
at loss time -> a_t). Architecturally identical to AdaLEAModel (shared
backbone family across all 3 baselines per RQ1 protocol); z_t IS the raw
logit (pre-sigmoid), matching the paper's notation exactly (z used
directly in FFR, sigmoid(z)=a used in BCE/AMC).
"""
import torch
import torch.nn as nn


class RiskPropModel(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()
        slow_r50 = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
        self.backbone = nn.ModuleList(slow_r50.blocks[:-1])
        self.feat_dim = 2048
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(self.feat_dim, 1)

    def forward_snippet(self, x):
        """x: (B, 3, 5, H, W) -> (B,) raw logit z."""
        for blk in self.backbone:
            x = blk(x)
        feat = self.gap(x).view(x.shape[0], self.feat_dim)
        feat = self.dropout(feat)
        return self.head(feat).squeeze(-1)

    def forward(self, x):
        """x: (B, N, 3, 5, H, W) sequence of N snippets, OR (B, 3, 5, H, W)
        a single snippet (eval path, matches TOP/AdaLEA's val loader shape).
        Returns (B, N) or (B,) logits respectively."""
        if x.dim() == 5:
            return self.forward_snippet(x)
        B, N = x.shape[0], x.shape[1]
        x = x.view(B * N, *x.shape[2:])
        z = self.forward_snippet(x)
        return z.view(B, N)
