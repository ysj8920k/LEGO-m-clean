from ultralytics import YOLO
import cv2
import math 
# start webcam

# model
"""
model = YOLO("Yolo-LEGO-test/runs/segment/train_Medium/weights/best.pt")

results=model(source=1,show=True,conf=0.3)

print(results)
"""

import cv2

cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
cap.set(3, 1920)
cap.set(4, 1080)

model = YOLO("Yolo-LEGO-test/runs/segment/train167/weights/best.pt")

while True:
    ret, img= cap.read()
    results = model.predict(source=img, show=True,conf=0.5)#, retina_masks=True)

    #cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()