from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv11 model
model = YOLO("best.pt")  # or 'yolo11n.pt' for pretrained

CONF_THRESHOLD = 0.25
IMG_SIZE = 640
PADDING = 10

def detect_plates(image):
    """
    Detect number plates in the given image using YOLOv11.
    Returns a list of cropped plate regions and bounding box info.
    """
    results = model.predict(source=image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
    boxes = []
    crops = []

    if len(results) > 0:
        r = results[0]
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        h, w = image.shape[:2]
        for (x1, y1, x2, y2), conf_val in zip(xyxy, confs):
            if conf_val >= CONF_THRESHOLD:
                x1p, y1p = max(0, int(x1 - PADDING)), max(0, int(y1 - PADDING))
                x2p, y2p = min(w - 1, int(x2 + PADDING)), min(h - 1, int(y2 + PADDING))
                plate_crop = image[y1p:y2p, x1p:x2p]
                boxes.append((x1p, y1p, x2p, y2p, conf_val))
                crops.append(plate_crop)
    return boxes, crops
