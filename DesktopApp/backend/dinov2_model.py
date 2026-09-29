import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

EMBEDDING_DIM = 256
IMG_SIZE = 518
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_top = (max_dim - h) // 2
        pad_right = max_dim - w - pad_left
        pad_bottom = max_dim - h - pad_top
        return T.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=128)

clean_transform = T.Compose([
    PadToSquare(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

aug_transform = T.Compose([
    PadToSquare(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.88, 1.12)),
    T.RandomPerspective(distortion_scale=0.2, p=0.3),
    T.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.30, hue=0.06),
    T.RandomGrayscale(p=0.08),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

class DINOv2DualStream(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Load local or hub DINOv2 ViT-B/14 backbone
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        from torch.utils.checkpoint import checkpoint as grad_ckpt
        for blk in self.backbone.blocks:
            original_fwd = blk.forward
            blk.forward = lambda x, _fwd=original_fwd: grad_ckpt(_fwd, x, use_reentrant=False)

        self.fusion_gate = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.Sigmoid()
        )

        self.projection_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, embedding_dim),
        )

    def _attn_patch_mean(self, x):
        feat = self.backbone.forward_features(x)
        patches = feat['x_norm_patchtokens']
        importance = patches.norm(dim=-1, keepdim=True)
        importance = importance / (importance.sum(1, keepdim=True) + 1e-8)
        return (patches * importance).sum(1)

    def forward_one(self, g, l):
        cls = self.backbone.forward_features(g)['x_norm_clstoken']
        loc = self._attn_patch_mean(l)
        
        concat_feats = torch.cat([cls, loc], dim=1)
        gate = self.fusion_gate(concat_feats)
        fused_feat = gate * cls + (1 - gate) * loc
        
        emb = self.projection_head(fused_feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, a, p, n):
        return self.forward_one(*a), self.forward_one(*p), self.forward_one(*n)
