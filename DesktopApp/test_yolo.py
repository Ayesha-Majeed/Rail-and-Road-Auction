import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO

yolo = YOLO("yolo11x.pt")
img_path = "/home/kk/Desktop/CV projects/Rail-and-Road-Auction-main/Full_Dataset_turn2/untrained classes/British Columbia Railway/bcr312.jpg"

img_pil_1 = Image.open(img_path).convert("RGB")
res1 = yolo(img_pil_1, verbose=False)[0]
if len(res1.boxes):
    boxes = res1.boxes.xyxy.cpu().numpy()
    classes = res1.boxes.cls.cpu().numpy().astype(int)
    mask = classes == 6
    if mask.any():
        for box in boxes[mask]:
            print("PIL Box:", box)

img_cv = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
img_pil_2 = Image.fromarray(img_rgb)
res2 = yolo(img_pil_2, verbose=False)[0]
if len(res2.boxes):
    boxes = res2.boxes.xyxy.cpu().numpy()
    classes = res2.boxes.cls.cpu().numpy().astype(int)
    mask = classes == 6
    if mask.any():
        for box in boxes[mask]:
            print("CV2 Box:", box)

