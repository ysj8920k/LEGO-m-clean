from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)
#cap.set(3, 512)
#cap.set(4, 512)

# model
model = YOLO("runs/pose/train5_150_epoch_v1/weights/best.pt")

# object classes
classNames = ['none','2x4','2x4p','2x2','2x2p','1x4','1x4p','1x2','1x2p','1x1','1x1p']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    if results != None:
        # coordinates
        for r in results:
            boxes = r.boxes
            keypoints=r.keypoints
            if boxes != None:
                print("found")
                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, classNames[cls]+': '+ str(confidence), org, font, fontScale, color, thickness)

                    if keypoints != None:
                        for keypoint in keypoints:
                            print(str(keypoints)+'keypoints print')
                            x, y = keypoint  # Replace with actual keypoint coordinates
                            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)  # Draw a red circle at each keypoint


                cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()