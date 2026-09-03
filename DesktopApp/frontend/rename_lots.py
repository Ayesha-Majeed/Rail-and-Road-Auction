import os
import random
from pathlib import Path

base_dir = Path("/home/kk/Downloads/test")

lot_id = 1000
total_renamed = 0
total_lots = 0

for subdir in base_dir.iterdir():
    if not subdir.is_dir():
        continue
    
    # Get all image files in this subdir
    files = [f for f in subdir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp']]
    
    if not files:
        continue
        
    random.shuffle(files)
    
    idx = 0
    while idx < len(files):
        # Pick a random lot size between 20 and 40 (as requested 20-30 or 50-60, we choose 20-40)
        lot_size = random.randint(20, 40)
        lot_files = files[idx : idx + lot_size]
        
        if not lot_files:
            break
            
        # Rename files in this lot
        for i, f in enumerate(lot_files, start=1):
            new_name = f"{lot_id}_{i:02d}{f.suffix}"
            new_path = f.with_name(new_name)
            f.rename(new_path)
            total_renamed += 1
            
        lot_id += 1
        total_lots += 1
        idx += lot_size

print(f"✅ Successfully processed!")
print(f"Total Lots Created: {total_lots}")
print(f"Total Files Renamed: {total_renamed}")
print(f"Lot IDs used: 1000 to {lot_id-1}")
