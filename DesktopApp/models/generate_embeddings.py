import os
import json
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image, ImageFile
from tqdm import tqdm

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True







# We import the model class and transforms from train_phase5.py
from train_phase5 import DINOv2DualStream, PadToSquare, clean_transform, EMBEDDING_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_val_embeddings(dataset_dirs, crops_json_path, model_weights_path, output_pt_path):
    print(f"Using device: {DEVICE}")
    print("Loading DINOv2 DualStream model...")
    model = DINOv2DualStream(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    
    print(f"Loading weights from {model_weights_path}...")
    ckpt = torch.load(model_weights_path, map_location=DEVICE, weights_only=True)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    
    print(f"Loading crops metadata from {crops_json_path}...")
    crops_dict = {}
    if os.path.exists(crops_json_path):
        with open(crops_json_path, 'r') as f:
            crops_dict = json.load(f)
    else:
        print(f"Warning: Crops JSON not found at {crops_json_path}. Will use full images.")
        
    embeddings = []
    labels = []
    
    for dataset_dir in dataset_dirs:
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset directory '{dataset_dir}' not found! Skipping...")
            continue
            
        split_name = os.path.basename(os.path.normpath(dataset_dir))
        classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
        print(f"\nProcessing directory '{dataset_dir}' (Split: {split_name})")
        print(f"Found {len(classes)} classes.")
        
        with torch.no_grad():
            for cls in tqdm(classes, desc=f"Processing {split_name} Classes"):
                cls_dir = os.path.join(dataset_dir, cls)
                for fname in os.listdir(cls_dir):
                    if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                        continue
                    if fname.lower().endswith("_loc.jpg"):
                        continue
                        
                    img_path = os.path.join(cls_dir, fname)
                    try:
                    img = Image.open(img_path).convert("RGB")
                    
                    global_img = img
                    local_img = img
                    
                    # Find crop info using relative path dynamically based on folder name
                    rel_path = f"{split_name}/{cls}/{fname}"
                    search_key = rel_path
                    if search_key not in crops_dict:
                        import re
                        clean_fname = re.sub(r'^(old_train_aug_\d+_|old_val_aug_\d+_|aug_\d+_)|(_aug_\d+)(?=\.\w+$)', '', fname)
                        search_key = f"{split_name}/{cls}/{clean_fname}"
                        
                    if search_key in crops_dict:
                        crop_info = crops_dict[search_key]
                        if "global_box" in crop_info:
                            g = crop_info["global_box"]
                            global_img = img.crop((g[0], g[1], g[2], g[3]))
                        if "local_box" in crop_info and "global_box" in crop_info:
                            l = crop_info["local_box"]
                            g = crop_info["global_box"]
                            abs_l = [g[0]+l[0], g[1]+l[1], g[0]+l[2], g[1]+l[3]]
                            local_img = img.crop((abs_l[0], abs_l[1], abs_l[2], abs_l[3]))
                            
                    if local_img.size[0] < 10 or local_img.size[1] < 10:
                        local_img = global_img
                        
                    g_tensor = clean_transform(global_img).unsqueeze(0).to(DEVICE)
                    l_tensor = clean_transform(local_img).unsqueeze(0).to(DEVICE)
                    
                    with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                        emb = model.forward_one(g_tensor, l_tensor)
                        
                    # Normalize embedding before saving just in case
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    
                    embeddings.append(emb.cpu().numpy()[0].tolist())
                    labels.append(cls)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print(f"\nGenerated {len(embeddings)} embeddings.")
    print(f"Saving to {output_pt_path}...")
    
    # Save as PyTorch binary format (.pt) for much faster loading and smaller file size
    torch.save({
        "embeddings": torch.tensor(embeddings, dtype=torch.float32),
        "labels": labels
    }, output_pt_path)
    
    print("✅ Successfully generated binary embeddings .pt file!")
    print("You can now place this 'val_embeddings.pt' inside your app's weights directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate val_embeddings.pt using Phase5 model")
    parser.add_argument("--dataset_dirs", type=str, nargs="+", required=True, help="Absolute paths to the dataset folders (e.g. /home/.../train /home/.../val)")
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CROPS_JSON = os.path.join(BASE_DIR, "crops.json")
    MODEL_WEIGHTS = os.path.join(BASE_DIR, "phase5_best.pth")
    OUTPUT_PT = os.path.join(BASE_DIR, "val_embeddings.pt")
    
    generate_val_embeddings(args.dataset_dirs, CROPS_JSON, MODEL_WEIGHTS, OUTPUT_PT)
