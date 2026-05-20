import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import json
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

# --- Paths & Config ---
DATASET_PATH = "/home/kk/Desktop/slides /Siamese_Ready_Dataset/train"
VAL_DATASET_PATH = "/home/kk/Desktop/slides /Siamese_Ready_Dataset/val"
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TripletRailroadDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = {}
        self.all_imgs = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir): continue
            paths = [os.path.join(cls_dir, f.replace('_glob.jpg', '')) 
                     for f in os.listdir(cls_dir) if f.endswith('_glob.jpg')]
            self.image_paths[cls] = paths
            self.all_imgs.extend([(cls, p) for p in paths])

    def __len__(self):
        return len(self.all_imgs)

    def get_dual_crops(self, base_path):
        """Loads pre-processed Global_crop, Local_crop, and coordinates"""
        try:
            glob_path = f"{base_path}_glob.jpg"
            loc_path = f"{base_path}_loc.jpg"
            geo_path = f"{base_path}_geo.json"
            
            global_crop = Image.open(glob_path).convert('RGB')
            local_crop = Image.open(loc_path).convert('RGB')
            
            with open(geo_path, 'r') as f:
                geo_data = json.load(f)
            
            y_norm = torch.tensor(geo_data[:4], dtype=torch.float32)
            o_norm = torch.tensor(geo_data[4:], dtype=torch.float32)
            
            return global_crop, local_crop, y_norm, o_norm
        except Exception as e:
            dummy = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (128, 128, 128))
            return dummy, dummy, torch.tensor([0,0,1,1], dtype=torch.float32), torch.tensor([0,0,1,1], dtype=torch.float32)

    def __getitem__(self, idx):
        anchor_cls, anchor_path = self.all_imgs[idx]
        pos_path = random.choice(self.image_paths[anchor_cls])
        while pos_path == anchor_path and len(self.image_paths[anchor_cls]) > 1:
            pos_path = random.choice(self.image_paths[anchor_cls])
        neg_cls = random.choice([c for c in self.classes if c != anchor_cls])
        neg_path = random.choice(self.image_paths[neg_cls])
        
        a_g, a_l, a_y, a_o = self.get_dual_crops(anchor_path)
        p_g, p_l, p_y, p_o = self.get_dual_crops(pos_path)
        n_g, n_l, n_y, n_o = self.get_dual_crops(neg_path)
        
        if self.transform:
            a_g, a_l = self.transform(a_g), self.transform(a_l)
            p_g, p_l = self.transform(p_g), self.transform(p_l)
            n_g, n_l = self.transform(n_g), self.transform(n_l)
            
        return (a_g, a_l, a_y, a_o), (p_g, p_l, p_y, p_o), (n_g, n_l, n_y, n_o)

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.geo_branch = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2048 + 2048 + 32, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, embedding_dim))

    def forward_one(self, img_glob, img_loc, boxes):
        f_g = self.feature_extractor(img_glob).view(img_glob.size(0), -1)
        f_l = self.feature_extractor(img_loc).view(img_loc.size(0), -1)
        f_geo = self.geo_branch(boxes)
        combined = torch.cat((f_g, f_l, f_geo), dim=1)
        return nn.functional.normalize(self.fusion(combined), p=2, dim=1)

    def forward(self, a, p, n):
        a_emb = self.forward_one(a[0], a[1], torch.cat((a[2], a[3]), dim=1))
        p_emb = self.forward_one(p[0], p[1], torch.cat((p[2], p[3]), dim=1))
        n_emb = self.forward_one(n[0], n[1], torch.cat((n[2], n[3]), dim=1))
        return a_emb, p_emb, n_emb

class TripletLoss(nn.Module):
    """Triplet loss with Online Hard Mining support"""
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        d_p = (anchor - positive).pow(2).sum(1)
        d_n = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(d_p - d_n + self.margin)
        
        mask = losses > 0
        num_hard = mask.sum().item()
        
        if num_hard > 0:
            loss = losses[mask].mean() if size_average else losses[mask].sum()
        else:
            loss = losses.mean()
        return loss, num_hard


from tqdm import tqdm

# --- Main Training ---
if __name__ == "__main__":
    train_transform = T.Compose([
        T.ToTensor(), 
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_ds = TripletRailroadDataset(DATASET_PATH, transform=train_transform)
    val_ds = TripletRailroadDataset(VAL_DATASET_PATH, transform=train_transform)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=8)

    model = SiameseNetwork().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = TripletLoss()
    scaler = torch.amp.GradScaler('cuda')

    # Full State Resume tracking
    checkpoint_path = "siamese_ocr_checkpoint.pth"
    best_path = "/home/kk/Desktop/slides /Siamese_OCR_Pipeline/siamese_best.pth"
    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(checkpoint_path):
        print(f"--- 🔄 Resuming from Last Checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from Epoch {start_epoch} | Previous Best Val Loss: {best_val_loss:.4f}")

    print(f"Starting Training on {DEVICE} (AMP Optimized)...")
    for epoch in range(start_epoch, 50):
        # --- Train Phase ---
        model.train()
        train_loss = 0
        total_hard = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50 [Train]")
        for i, (a, p, n) in enumerate(pbar):
            a, p, n = [ [x.to(DEVICE) for x in triplet] for triplet in [a, p, n] ]
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                ae, pe, ne = model(a, p, n)
                loss, hard_count = criterion(ae, pe, ne)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            total_hard += hard_count
            
            # Update progress bar
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'hard': f"{hard_count}/2",
                'lr': f"{curr_lr:.1e}"
            })
        
        # --- Val Phase ---
        model.eval()
        val_loss = 0
        torch.cuda.empty_cache()
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/50 [Val]")
        with torch.no_grad():
            for i, (a, p, n) in enumerate(val_pbar):
                a, p, n = [ [x.to(DEVICE) for x in triplet] for triplet in [a, p, n] ]
                with torch.amp.autocast('cuda'):
                    ae, pe, ne = model(a, p, n)
                    v_loss, _ = criterion(ae, pe, ne)
                val_loss += v_loss.item()
                val_pbar.set_postfix({'val_loss': f"{v_loss.item():.4f}"})
        
        scheduler.step()
        avg_train = train_loss/len(train_loader)
        avg_val = val_loss/len(val_loader)
        
        print(f"\n--- Epoch {epoch+1} Results ---")
        print(f"Avg Train Loss: {avg_train:.4f} | Avg Val Loss: {avg_val:.4f} | Total Hard: {total_hard}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_path)
            print(f"⭐ New Best Model Weights Saved!")
        
        # Save Full Checkpoint for Resuming
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }, checkpoint_path)
        
        torch.cuda.empty_cache()
        print(f"---------------------------------\n")
