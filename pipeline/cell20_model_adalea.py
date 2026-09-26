import torch
import torch.nn as nn


class AdaLEAModel(nn.Module):
    """
    AdaLEA risk model, adapted to the shared RQ1 protocol (single causal
    5-frame window -> single risk score), matched-budget backbone with
    TOP/RiskProp (SlowOnly-R50, Kinetics-400 pretrained).

    Architecture:
        SlowOnly-R50 backbone -> AdaptiveAvgPool3d(1) -> Dropout(0.2)
        -> Linear(2048, 1) -> logit (sigmoid applied by loss/eval code)

    Input : (B, 3, 5, 224, 224)  -- proposal Muc 5.5: causal 5-frame snippet
    Output: (B,) raw logit
    """

    def __init__(self, dropout_p=0.2):
        super().__init__()
        slow_r50 = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'slow_r50', pretrained=True)
        self.backbone = nn.ModuleList(slow_r50.blocks[:-1])
        self.feat_dim = 2048
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(self.feat_dim, 1)

    def forward(self, x):
        for blk in self.backbone:
            x = blk(x)
        feat = self.gap(x).view(x.shape[0], self.feat_dim)
        feat = self.dropout(feat)
        return self.head(feat).squeeze(-1)
