import cv2

cap = cv2.VideoCapture(0)
#cap.set(3, 512)
#cap.set(4, 512)


while True:
    succes, img= cap.read()

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()