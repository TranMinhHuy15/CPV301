import torch
import torch.nn as nn


class TOPModel(nn.Module):
    """
    TOP baseline model (Zhao et al., NeurIPS 2025), adapted to the shared
    RQ1 protocol (single causal 5-frame window -> 20-horizon cumulative
    risk scores), matched-budget backbone with AdaLEA/RiskProp
    (SlowOnly-R50, Kinetics-400 pretrained).

    Each of the 20 output logits h (0-indexed) corresponds to the
    cumulative question "will the collision happen within (h+1)*0.1
    seconds from now?" -- bin width 0.1s, covering 0.1s..2.0s. Matches
    cell16_eval_cached_5f.py's LEAD_TO_HIDX ({0.5: 4, 1.0: 9, 1.5: 14}).

    Input : (B, 3, 5, 224, 224)
    Output: (B, 20) raw logits
    """

    NUM_HORIZONS = 20

    def __init__(self, dropout_p=0.2):
        super().__init__()
        slow_r50 = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'slow_r50', pretrained=True)
        self.backbone = nn.ModuleList(slow_r50.blocks[:-1])
        self.feat_dim = 2048
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = nn.Linear(self.feat_dim, self.NUM_HORIZONS)

    def forward(self, x):
        for blk in self.backbone:
            x = blk(x)
        feat = self.gap(x).view(x.shape[0], self.feat_dim)
        feat = self.dropout(feat)
        return self.head(feat)   # (B, 20)
