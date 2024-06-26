from ultralytics import YOLO
import cv2
import random
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import pandas as pd
import MV_Main_belt

#import RV_Math 

def calc_length(x,y):
    l1 = np.sqrt((x[1] - x[0])**2 + (y[0] - y[1])**2)
    return l1

def calibration(webcam):
    Calibration_bricks = MV_Main_belt.main_mv_belt(300,webcam)

    x1=Calibration_bricks['x-Cordinates [px]'].values[-1]
    y1=Calibration_bricks['y-Cordinates [px]'].values[-1]
    C_1=Calibration_bricks['Angle [rad]'].values[-1]

    x2=Calibration_bricks['x-Cordinates [px]'].values[-2]
    y2=Calibration_bricks['y-Cordinates [px]'].values[-2]
    C_2=Calibration_bricks['Angle [rad]'].values[-2]
    #0,711y
    #0,710x

    result_x=[x1,x2]
    result_y=[y1,y2] 

    result_angle=[C_1,C_2]

    #robot cords

    xy_upper=[328.06,-144.35]
    xy_lower=[546.00,155.87]

    length_mm=calc_length(xy_upper,xy_lower)
    length_px=calc_length(result_x,result_y)
    mm_pr_px=length_mm/length_px
    T_to_brick_x=100 #in mm
    T_to_brick_y=100 #in mm
    i_min=0
    T=[T_to_brick_x-result_x[i_min]*mm_pr_px,T_to_brick_y-result_y[i_min]*mm_pr_px, np.mean(result_angle)]

    print('test')
    print(result_x)
    print(result_y)
    print('The average angle of the brick is: '+str(np.mean(result_angle)))
    print('The mm per pixel is: ' + str(mm_pr_px))
    print('The first found brick had the following coordinates [x,y] in px: ' + str([result_x[i_min],result_y[i_min]]))
    print('The first found brick had the following offset from [0,0] [x,y] in mm: ' + str([result_x[i_min]*mm_pr_px,result_y[i_min]*mm_pr_px]))
    print('To [0,0] then has the following transformation [x,y,C]: '+str(T))
    ROI=[0,-1,y1+25*mm_pr_px,y2-25*mm_pr_px]
    #px to mm 

    return T, ROI

if __name__ == '__main__':
    webcam = cv2.VideoCapture(1)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -6 )

    calibration(webcam)


