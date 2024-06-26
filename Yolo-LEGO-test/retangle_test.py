import cv2
import numpy as np 
#https://namkeenman.wordpress.com/2015/12/18/open-cv-determine-angle-of-rotatedrect-minarearect/
#https://stackoverflow.com/questions/34237253/detect-centre-and-angle-of-rectangles-in-an-image-using-opencv



img = cv2.imread("test2.png",0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh,1,2)



for contour in contours:
    area = cv2.contourArea(contour)
    if area>100000:
        contours.remove(contour)




cnt = contours[0]

epsilon = 0.02*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)

print ('No of rectangles',len(approx))


#finding the centre of the contour
M = cv2.moments(cnt)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

print (cx,cy)

cv2.imshow('output', img)
cv2.waitKey()