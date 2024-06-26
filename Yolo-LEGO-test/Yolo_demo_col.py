from ultralytics import YOLO
import cv2
import math 
# start webcam

# model
model = YOLO("runs/detect/train10_farve_100_epocher/weights/best.pt")

results=model(source=1,show=True,conf=0.3)