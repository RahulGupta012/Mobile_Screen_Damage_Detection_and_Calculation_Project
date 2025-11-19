
import cv2
import numpy as np
import os
from ultralytics import YOLO
import torch
from torchvision import models, transforms



# Defining the class for Mobile img Detection
class MobileDetector:
    def __init__(self, img_data):
        model_path = 'mobile_detector.pt'  
        self.model = YOLO(model_path)
        self.img_data = img_data

    def detected_objects(self):
        results = self.model.predict(self.img_data, verbose=False)
        cropped_images = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = self.img_data[y1:y2, x1:x2]
                cropped_images.append(cropped_img)
        return cropped_images




# Making a pipline which takes the img as input and gives the segmented img
# Segmented img go to the edge detector to calculate the damage percentage
def damage_pipeline(img_data):
    detector = MobileDetector(img_data)
    cropped_images = detector.detected_objects()
    if not cropped_images:
        return None, None, 0.0  # No phone detected
    cropped_img = cropped_images[0]

    # Load segmentation model from torchvision
    model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    transform = transforms.Compose([transforms.ToTensor()])

    # Convert image to tensor
    if cropped_img.ndim == 3 and cropped_img.shape[2] == 3:
        rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    elif cropped_img.ndim == 3 and cropped_img.shape[2] == 4:
        rgb_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGRA2RGB)
    else:
        rgb_img = cropped_img
    img_tensor = transform(rgb_img)

   #make prediction
    model.eval()
    with torch.no_grad():
        output = model([img_tensor])[0]

    masks = (output["masks"] > 0.5).squeeze().cpu().numpy()
    rgb_img_disp = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    if masks.ndim == 3 and masks.shape[0] > 0:
        mask = masks[0]
    elif masks.ndim == 2:
        mask = masks
    else:
        mask = np.zeros(rgb_img_disp.shape[:2], dtype=bool)

    # Apply mask to get segmented object
    segmented = np.zeros_like(rgb_img_disp, dtype=np.uint8)
    segmented[mask] = rgb_img_disp[mask]

    # Edge detection for damage
    gray = cv2.cvtColor(segmented, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    edge_pixels = np.sum(edges > 0)
    total_pixels = gray.size
    damage_percent = (edge_pixels / total_pixels) * 100

    return segmented, edges, damage_percent