"""
DINOv2 Railroad — Dual-Stream Fine-tuning
==========================================
Phase 5: Linear Probing (Permanently Frozen Backbone)

What changed vs Phase 4:
  - DINOv2 backbone is permanently frozen (as per Senior's advice).
  - Starts from phase4_best.pth (can be pointed to phase3_best.pth if needed).
  - ARCFACE_WEIGHT reduced to 0.3.
  - Dropout increased to 0.5.
  - Weight Decay increased to 1e-3.
"""

import os
import json
import socket
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm
import wandb

def _has_internet(host="api.wandb.ai", port=443, timeout=5):
    try:
        socket.setdefaulttimeout(timeout)
        with socket.create_connection((host, port)):
            return True
    except OSError:
        return False

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_DIR       = r"F:\Ayesha\Rail_Road_Auctions\RailRoad_Split_Dataset\RailRoad_Split_Dataset"
SAVE_DIR        = r"F:\Ayesha\Rail_Road_Auctions\DINOv2_Railroad_Pipeline"
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoints", "phase5_checkpoint.pth")
BEST_PATH       = os.path.join(SAVE_DIR, "checkpoints", "phase5_best.pth")
LOG_PATH        = os.path.join(SAVE_DIR, "logs", "phase5_training_log.json")
BASE_MODEL      = os.path.join(SAVE_DIR, "checkpoints", "phase4_best.pth") 

WANDB_PROJECT   = "DINOv2-Railroad"
WANDB_ENTITY    = "mlbenchpvtltd-ml-bench"

os.makedirs(os.path.join(SAVE_DIR, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "logs"),        exist_ok=True)

EPOCHS          = 150
PATIENCE        = 30
BATCH_SIZE      = 2
ACCUMULATION_STEPS = 8
TRIPLET_WEIGHT  = 1.0
ARCFACE_WEIGHT  = 0.3
IMG_SIZE        = 518
EMBEDDING_DIM   = 256
MARGIN          = 0.3
LR_HEAD         = 1e-4
WEIGHT_DECAY    = 1e-3
HARD_MINE_BATCH = 4
MINE_BATCH_SIZE = 4

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

clean_transform = T.Compose([
    PadToSquare(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

import re
AUG_REGEX = re.compile(r'^(old_train_aug_\d+_|old_val_aug_\d+_|aug_\d+_)|(_aug_\d+)(?=\.\w+$)')

class DualStreamDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform  = transform
        self.split      = split
        self.samples    = []
        
        self.crops_dict = {}
        crops_json_path = r"F:\Ayesha\Rail_Road_Auctions\DINOv2_Railroad_Pipeline\crops.json"
        if os.path.exists(crops_json_path):
            with open(crops_json_path, 'r') as f:
                self.crops_dict = json.load(f)

        split_dir = os.path.join(SPLIT_DIR, split)
        if not os.path.exists(split_dir):
            return

        train_dir_path = os.path.join(SPLIT_DIR, "train")
        class_counts = {}
        for c in os.listdir(train_dir_path):
            cp = os.path.join(train_dir_path, c)
            if os.path.isdir(cp):
                orig_cnt = len([f for f in os.listdir(cp) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and '_aug' not in f])
                class_counts[c] = orig_cnt

        sorted_top50 = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        self.classes      = sorted([x[0] for x in sorted_top50])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(split_dir, cls)
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    if f.lower().endswith("_loc.jpg"):
                        continue
                    
                    full_path = os.path.join(cls_dir, f)
                    rel_path = f"{split}/{cls}/{f}"
                    
                    search_key = rel_path
                    if search_key not in self.crops_dict:
                        clean_fname = AUG_REGEX.sub('', f)
                        target_rel_path = f"{split}/{cls}/{clean_fname}"
                        if target_rel_path in self.crops_dict:
                            search_key = target_rel_path

                    if search_key in self.crops_dict:
                        self.samples.append((full_path, search_key, self.class_to_idx[cls]))

        self.labels = np.array([s[2] for s in self.samples])

        class_counts = np.bincount(self.labels)
        class_weights = 1.0 / (class_counts + 1e-5)
        sample_weights = class_weights[self.labels]
        self.sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_path, search_key, label = self.samples[idx]
        try:
            img = Image.open(full_path).convert("RGB")
            
            global_img = img
            local_img  = img
            
            if search_key in self.crops_dict:
                crop_info = self.crops_dict[search_key]
                if "global_box" in crop_info:
                    g_box = crop_info["global_box"]
                    global_img = img.crop((g_box[0], g_box[1], g_box[2], g_box[3]))
                if "local_box" in crop_info and "global_box" in crop_info:
                    l_box = crop_info["local_box"]
                    g_box = crop_info["global_box"]
                    abs_l = [
                        g_box[0] + l_box[0],
                        g_box[1] + l_box[1],
                        g_box[0] + l_box[2],
                        g_box[1] + l_box[3],
                    ]
                    local_img = img.crop((abs_l[0], abs_l[1], abs_l[2], abs_l[3]))

        except Exception as e:
            global_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            local_img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (128, 128, 128))

        if local_img.size[0] < 10 or local_img.size[1] < 10:
            local_img = global_img

        tf = self.transform if self.transform else clean_transform
        return tf(global_img), tf(local_img), label, idx


class TripletDataset(Dataset):
    def __init__(self, base_dataset, triplets):
        self.base     = base_dataset
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_i, p_i, n_i = self.triplets[idx]
        a_g, a_l, a_lbl, _ = self.base[a_i]
        p_g, p_l, p_lbl, _ = self.base[p_i]
        n_g, n_l, n_lbl, _ = self.base[n_i]
        return (a_g, a_l, a_lbl), (p_g, p_l, p_lbl), (n_g, n_l, n_lbl)

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine = torch.clamp(cosine, -0.9999, 0.9999)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class DINOv2DualStream(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        print("  Loading DINOv2 ViT-B/14 backbone...")
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
        feat         = self.backbone.forward_features(x)
        patches      = feat['x_norm_patchtokens']              
        importance   = patches.norm(dim=-1, keepdim=True)       
        importance   = importance / (importance.sum(1, keepdim=True) + 1e-8)
        return (patches * importance).sum(1)                    

    def forward_one(self, g, l):
        cls  = self.backbone.forward_features(g)['x_norm_clstoken']
        loc  = self._attn_patch_mean(l)
        
        concat_feats = torch.cat([cls, loc], dim=1)
        gate = self.fusion_gate(concat_feats)
        fused_feat = gate * cls + (1 - gate) * loc
        
        emb  = self.projection_head(fused_feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, a, p, n):
        return self.forward_one(*a), self.forward_one(*p), self.forward_one(*n)


def get_hard_triplets(model, dataset, device, margin=MARGIN, batch_size=32):
    model.eval()
    all_embs, all_labels = [], []

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=getattr(dataset, 'sampler', None),
        shuffle=(not hasattr(dataset, 'sampler')),
        num_workers=min(4, os.cpu_count()),
        pin_memory=True
    )

    with torch.no_grad():
        for g, l, labels, _ in tqdm(loader, desc="Generating Embeddings", leave=False):
            with torch.amp.autocast('cuda'):
                emb = model.forward_one(g.to(device), l.to(device))
            all_embs.append(emb)
            all_labels.extend(labels)

    all_embs = torch.cat(all_embs, dim=0)                
    all_labels = torch.tensor(all_labels, device='cpu')   

    N = all_embs.size(0)

    chunk_size = 2000
    hard_pos_idx_list = []
    hard_pos_dists_list = []
    hard_neg_idx_list = []
    hard_neg_dists_list = []

    with torch.no_grad():
        for start_i in range(0, N, chunk_size):
            end_i = min(start_i + chunk_size, N)
            chunk_embs = all_embs[start_i:end_i] 
            
            chunk_dists = torch.cdist(chunk_embs, all_embs, p=2)

            chunk_labels = all_labels[start_i:end_i].unsqueeze(1) 
            all_labels_row = all_labels.unsqueeze(0)              

            labels_equal = (chunk_labels == all_labels_row).to(device)

            pos_mask = labels_equal.clone()
            for r_idx in range(end_i - start_i):
                pos_mask[r_idx, start_i + r_idx] = False

            neg_mask = ~labels_equal

            pos_dists = chunk_dists.masked_fill(~pos_mask, -float('inf'))
            hp_dists, hp_idx = pos_dists.max(dim=1)

            neg_dists = chunk_dists.masked_fill(~neg_mask, float('inf'))
            hn_dists, hn_idx = neg_dists.min(dim=1)

            hard_pos_dists_list.append(hp_dists.cpu())
            hard_pos_idx_list.append(hp_idx.cpu())
            hard_neg_dists_list.append(hn_dists.cpu())
            hard_neg_idx_list.append(hn_idx.cpu())

            del chunk_dists, pos_mask, neg_mask, pos_dists, neg_dists
            torch.cuda.empty_cache()

    hard_pos_dists = torch.cat(hard_pos_dists_list)
    hard_pos_idx = torch.cat(hard_pos_idx_list)
    hard_neg_dists = torch.cat(hard_neg_dists_list)
    hard_neg_idx = torch.cat(hard_neg_idx_list)

    triplet_mask = (hard_neg_dists - hard_pos_dists < margin) & (hard_pos_dists != -float('inf')) & (hard_neg_dists != float('inf'))

    valid_anchors = torch.where(triplet_mask)[0].tolist()
    hard_pos_idx_l = hard_pos_idx.tolist()
    hard_neg_idx_l = hard_neg_idx.tolist()

    triplets = [(i, hard_pos_idx_l[i], hard_neg_idx_l[i]) for i in valid_anchors]

    if len(triplets) == 0:
        all_labels_np = all_labels.cpu().numpy()
        for i in range(N):
            anc_lbl = all_labels_np[i]
            pos_idx = np.where(all_labels_np == anc_lbl)[0]
            pos_idx = pos_idx[pos_idx != i]
            neg_idx = np.where(all_labels_np != anc_lbl)[0]
            if len(pos_idx) > 0 and len(neg_idx) > 0:
                triplets.append((i, int(np.random.choice(pos_idx)), int(np.random.choice(neg_idx))))

    return triplets


def main():
    global BATCH_SIZE, HARD_MINE_BATCH, MINE_BATCH_SIZE

    wandb_mode = "online" if _has_internet() else "offline"
    print(f"  WandB mode   : {wandb_mode}" + (" (no internet — logs saved locally)" if wandb_mode == "offline" else ""))
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="Phase5_FrozenBackbone",
        mode=wandb_mode,
        config={
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "triplet_weight": TRIPLET_WEIGHT,
            "arcface_weight": ARCFACE_WEIGHT,
            "img_size": IMG_SIZE,
            "embedding_dim": EMBEDDING_DIM,
            "margin": MARGIN,
            "lr_head": LR_HEAD
        }
    )

    print("\nBuilding Phase 5 datasets...")
    train_clean = DualStreamDataset("train", transform=clean_transform)
    train_aug   = DualStreamDataset("train", transform=aug_transform)
    val_clean   = DualStreamDataset("val",   transform=clean_transform)

    print(f"  Train : {len(train_clean):,} samples | {len(train_clean.classes)} classes")
    print(f"  Val   : {len(val_clean):,}   samples")

    with open(os.path.join(SAVE_DIR, "class_mapping_phase5.json"), "w") as f:
        json.dump(train_clean.class_to_idx, f, indent=2)

    print("\nInitializing model...")
    model = DINOv2DualStream(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    if os.path.exists(BASE_MODEL):
        print(f"✅ SUCCESSFULLY LOADED BASE WEIGHTS FROM: {BASE_MODEL}")
        ckpt_base = torch.load(BASE_MODEL, map_location=DEVICE)
        
        if 'model_state_dict' in ckpt_base:
            model.load_state_dict(ckpt_base['model_state_dict'], strict=False)
        else:
            model.load_state_dict(ckpt_base, strict=False)
            
    else:
        print(f"⚠️ Warning: Base model not found at {BASE_MODEL}")
    
    num_classes = len(train_clean.classes)
    arcface = ArcFace(in_features=EMBEDDING_DIM, out_features=num_classes, s=30.0, m=0.50).to(DEVICE)

    # Only optimize Projection Head and Fusion Gate! 
    # The backbone parameters are NOT passed to the optimizer.
    optimizer = optim.AdamW([
        {'params': model.fusion_gate.parameters(),      'lr': LR_HEAD},
        {'params': model.projection_head.parameters(),  'lr': LR_HEAD},
        {'params': arcface.parameters(),                'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    criterion_triplet = nn.TripletMarginLoss(margin=MARGIN, p=2)
    criterion_ce = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda')

    start_epoch   = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    training_log  = []

    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n🔄 Resuming from Phase 5 checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        if 'arcface_state_dict' in ckpt:
            arcface.load_state_dict(ckpt['arcface_state_dict'])
            
        start_epoch   = ckpt['epoch']
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        training_log  = ckpt.get('training_log', [])
        print(f"  ↳ Epoch {start_epoch} | Best Val Loss: {best_val_loss:.4f} | Patience: {epochs_no_improve}/{PATIENCE}")
    else:
        print("  Starting fresh Phase 5 training.")

    print(f"\n🚀 DINOv2 Phase 5 (Frozen Backbone) | {EPOCHS} Epochs | {DEVICE}")
    print(f"   Image Size  : {IMG_SIZE}×{IMG_SIZE}")
    print(f"   Batch Size  : {BATCH_SIZE} x {ACCUMULATION_STEPS} (Accum) = {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"   Mining Batch: {HARD_MINE_BATCH}")
    print(f"   LR head     : {LR_HEAD:.0e}\n")

    epoch = start_epoch
    while epoch < EPOCHS:
        try:
            print(f"⛏  Epoch {epoch+1}/{EPOCHS} — Mining hard triplets... (GPU Vectorized, bs={MINE_BATCH_SIZE})")
            train_triplets = get_hard_triplets(model, train_clean, DEVICE, batch_size=MINE_BATCH_SIZE)
            val_triplets   = get_hard_triplets(model, val_clean,   DEVICE, batch_size=MINE_BATCH_SIZE)
            print(f"   Triplets → Train: {len(train_triplets):,} | Val: {len(val_triplets):,}")

            if len(train_triplets) == 0:
                print("✨ No hard triplets — perfect class separation! Training done early.")
                break

            anchor_labels  = [train_clean.labels[t[0]] for t in train_triplets]
            class_counts   = np.bincount(anchor_labels, minlength=len(train_clean.classes))
            class_weights  = 1.0 / np.maximum(class_counts, 1)
            sample_weights = [float(class_weights[l]) for l in anchor_labels]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            n_workers = min(4, os.cpu_count())
            train_loader = DataLoader(TripletDataset(train_aug,   train_triplets),
                                      batch_size=BATCH_SIZE, sampler=sampler,
                                      num_workers=n_workers, pin_memory=True)
            val_loader   = DataLoader(TripletDataset(val_clean, val_triplets),
                                      batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=n_workers, pin_memory=True)

            # ── PERMANENTLY FROZEN BACKBONE ──────────────────────────────────
            model.train()
            model.backbone.eval()
            for param in model.backbone.parameters():
                param.requires_grad = False
            
            mode_str = f"🔒 Stage 2: Linear Probing (Backbone is Permanently Frozen)"
            print(f"\n{mode_str}")

            train_loss = 0.0
            train_triplet_sum = 0.0
            train_arc_sum = 0.0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
            
            optimizer.zero_grad()

            for step, ((a_g, a_l, a_lbl), (p_g, p_l, p_lbl), (n_g, n_l, n_lbl)) in pbar:
                a_g, a_l, a_lbl = a_g.to(DEVICE), a_l.to(DEVICE), a_lbl.to(DEVICE)
                p_g, p_l, p_lbl = p_g.to(DEVICE), p_l.to(DEVICE), p_lbl.to(DEVICE)
                n_g, n_l, n_lbl = n_g.to(DEVICE), n_l.to(DEVICE), n_lbl.to(DEVICE)

                with torch.amp.autocast('cuda'):
                    a_e, p_e, n_e = model((a_g, a_l), (p_g, p_l), (n_g, n_l))
                    
                    loss_triplet = criterion_triplet(a_e, p_e, n_e)
                    
                    a_out = arcface(a_e, a_lbl)
                    p_out = arcface(p_e, p_lbl)
                    n_out = arcface(n_e, n_lbl)
                    loss_arc = (criterion_ce(a_out, a_lbl) + criterion_ce(p_out, p_lbl) + criterion_ce(n_out, n_lbl)) / 3.0
                    
                    loss = (TRIPLET_WEIGHT * loss_triplet) + (ARCFACE_WEIGHT * loss_arc)
                    loss_accum = loss / ACCUMULATION_STEPS

                scaler.scale(loss_accum).backward()

                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(arcface.parameters(), max_norm=1.0)
    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item()
                train_triplet_sum += loss_triplet.item()
                train_arc_sum += loss_arc.item()
                
                pbar.set_postfix(
                    total=f"{loss.item():.4f}",
                    trip=f"{loss_triplet.item():.4f}",
                    arc=f"{loss_arc.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                )

            avg_train = train_loss / len(train_loader)
            avg_train_triplet = train_triplet_sum / len(train_loader)
            avg_train_arc = train_arc_sum / len(train_loader)

            model.eval()
            val_loss = 0.0
            val_triplet_sum = 0.0
            val_arc_sum = 0.0
            torch.cuda.empty_cache()

            with torch.no_grad():
                vbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
                for (a_g, a_l, a_lbl), (p_g, p_l, p_lbl), (n_g, n_l, n_lbl) in vbar:
                    a_g, a_l, a_lbl = a_g.to(DEVICE), a_l.to(DEVICE), a_lbl.to(DEVICE)
                    p_g, p_l, p_lbl = p_g.to(DEVICE), p_l.to(DEVICE), p_lbl.to(DEVICE)
                    n_g, n_l, n_lbl = n_g.to(DEVICE), n_l.to(DEVICE), n_lbl.to(DEVICE)
                    with torch.amp.autocast('cuda'):
                        a_e, p_e, n_e = model((a_g, a_l), (p_g, p_l), (n_g, n_l))
                        
                        loss_triplet = criterion_triplet(a_e, p_e, n_e)
                        a_out = arcface(a_e, a_lbl)
                        p_out = arcface(p_e, p_lbl)
                        n_out = arcface(n_e, n_lbl)
                        loss_arc = (criterion_ce(a_out, a_lbl) + criterion_ce(p_out, p_lbl) + criterion_ce(n_out, n_lbl)) / 3.0
                        
                        loss = (TRIPLET_WEIGHT * loss_triplet) + (ARCFACE_WEIGHT * loss_arc)
                        
                        val_loss += loss.item()
                        val_triplet_sum += loss_triplet.item()
                        val_arc_sum += loss_arc.item()
                        
                        vbar.set_postfix(
                            total=f"{loss.item():.4f}",
                            trip=f"{loss_triplet.item():.4f}",
                            arc=f"{loss_arc.item():.4f}"
                        )

            avg_val = val_loss / len(val_loader) if val_loader else 0.0
            avg_val_triplet = val_triplet_sum / len(val_loader) if val_loader else 0.0
            avg_val_arc = val_arc_sum / len(val_loader) if val_loader else 0.0
            scheduler.step()

            print(f"\n{'─'*45}")
            print(f"  Epoch {epoch+1:03d}/{EPOCHS}")
            print(f"  Train Loss   : {avg_train:.4f} (Trip: {avg_train_triplet:.4f} | Arc: {avg_train_arc:.4f})")
            print(f"  Val Loss     : {avg_val:.4f} (Trip: {avg_val_triplet:.4f} | Arc: {avg_val_arc:.4f})")
            print(f"  Hard Triplets: {len(train_triplets):,}")
            print(f"  Batch Size   : {BATCH_SIZE}  |  Mining: {HARD_MINE_BATCH}")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train,
                "train_triplet_loss": avg_train_triplet,
                "train_arc_loss": avg_train_arc,
                "val_loss": avg_val,
                "val_triplet_loss": avg_val_triplet,
                "val_arc_loss": avg_val_arc,
                "lr_head": optimizer.param_groups[0]['lr'],
                "hard_triplets": len(train_triplets)
            })

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                epochs_no_improve = 0
                torch.save(model.state_dict(), BEST_PATH)
                print(f"  ⭐ New Best Model Saved! (Val Loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"  ⚠️ No improvement in Val Loss. Patience: {epochs_no_improve}/{PATIENCE}")

            training_log.append({
                'epoch':      epoch + 1,
                'train_loss': round(avg_train, 4),
                'val_loss':   round(avg_val,   4),
                'hard_trips': len(train_triplets),
                'batch_size': BATCH_SIZE,
            })

            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict':    scaler.state_dict(),
                'arcface_state_dict':   arcface.state_dict(),
                'best_val_loss':        best_val_loss,
                'epochs_no_improve':    epochs_no_improve,
                'training_log':         training_log,
            }, CHECKPOINT_PATH)

            if epochs_no_improve >= PATIENCE:
                print(f"\n🛑 Early stopping triggered! Validation loss did not improve for {PATIENCE} epochs.")
                break

            with open(LOG_PATH, "w") as f:
                json.dump(training_log, f, indent=2)

            torch.cuda.empty_cache()
            print(f"{'─'*45}\n")

            epoch += 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n⚠️  CUDA OOM on Epoch {epoch+1}!")
                if BATCH_SIZE > 1 or MINE_BATCH_SIZE > 8:
                    if BATCH_SIZE > 1:
                        BATCH_SIZE = max(1, BATCH_SIZE // 2)
                    if MINE_BATCH_SIZE > 8:
                        MINE_BATCH_SIZE = max(8, MINE_BATCH_SIZE // 2)
                    print(f"📉  Auto-reduced → Batch: {BATCH_SIZE} | Mining Batch: {MINE_BATCH_SIZE}")
                    print(f"🔄  Retrying Epoch {epoch+1}...\n")
                    continue
                else:
                    print("❌  Minimum batch size reached (1). Consider reducing IMG_SIZE.")
                    raise
            raise

    print("\n🎉 Phase 5 Training Complete!")
    print(f"   Best Val Loss : {best_val_loss:.4f}")
    print(f"   Best Model    : {BEST_PATH}")
    print(f"\n➡️  Next: Run build_faiss.py using phase5_best.pth for improved FAISS Vector Database!")

if __name__ == "__main__":
    main()
