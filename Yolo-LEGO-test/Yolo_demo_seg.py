from ultralytics import YOLO
import cv2
import math 
# start webcam

# model
model = YOLO("Yolo-LEGO-test/runs/segment/train167/weights/best.pt")

results=model.predict(source=1,show=True,retina_masks=True,conf=0.3)

print(results)