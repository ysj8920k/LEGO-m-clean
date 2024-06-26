from ultralytics import YOLO
import cv2
import random
import numpy as np
from math import atan2, cos, sin, sqrt, pi, radians
import math
import pandas as pd
import pyrealsense2 as rs
import json


def get_xyA(frames,webcam,roi,profile, calib_depth):
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    avg_center_list_x=[]
    avg_center_list_y=[]
    avg_of_angle=[]
    list_of_x=[]
    list_of_y=[]
    list_of_angle=[]
    list_of_type=[]
    list_of_colour=[]
    Cur_brick_type=[]
    Cur_brick_colour=[]

    std_x=[]
    std_y=[]
    std_angle=[]
    list_of_depth=[]
    avg_of_depth=[]



    Found=False
    t=0
    print('Package Imported')


    # Load a model
    model_type = YOLO("Yolo-LEGO-test/runs/segment/train133/weights/best.pt")
    model_colour = YOLO("Yolo-LEGO-test/runs/detect/train19/weights/best.pt")
    class_names_type = model_type.names
    class_names_model_colour = model_colour.names
    print('Class Names: ', class_names_type)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names_model_colour]
    # cap = cv2.VideoCapture('test.mp4')

    class_names_colour = model_colour.names
    print('Class Names: ', class_names_colour)




    while True:

        frameset = webcam.wait_for_frames()

        depth_image,depth_image_raw ,img=post_process(frameset)
        # Read the CSV file into a DataFrame
        # Load mtx from CSV
        mtx = np.genfromtxt('camera_matrix.csv', delimiter=',')

        # Load dist from CSV
        dist = np.genfromtxt('distortion_coefficients.csv', delimiter=',')




        #img = undistort_image(img, mtx, dist)
        #depth_image = undistort_image(depth_image, mtx, dist)
        #depth_image_raw = undistort_image(depth_image_raw, mtx, dist)

        h, w, _ = img.shape
        #results = model.predict(img, stream=True)
        results_type = model_type.track(img, stream=True, verbose=False)
        results_colour = model_colour(img, stream=True, verbose=False)
        for r in results_type:
            boxes_type = r.boxes  # Boxes object for bbox outputs
            masks_type = r.masks  # Masks object for segment masks outputs
            probs_type = r.probs  # Class probabilities for classification outputs

        for r in results_colour:
            boxes_colour = r.boxes  # Boxes object for bbox outputs
            # Masks object for segment masks outputs
            probs_colour = r.probs  # Class probabilities for classification output


        ####Moddellen "Tror på" segmenterings klodser 
        if boxes_colour is not None:
            #masks_colour = masks_colour.data.cpu()
            for box in boxes_colour:

                xmin = int(box.data[0][0])
                ymin = int(box.data[0][1])
                xmax = int(box.data[0][2])
                ymax = int(box.data[0][3])
                
                
                New_brick=True

                center=np.zeros(2)
                brick_type = 1                   
                #print('LEGO is found')

                center[0]=(xmax-xmin)/2+xmin
                center[1]=(ymax-ymin)/2+ymin
                
                colour_type=box.cls

                if float(box.conf)>=0.5:
                    plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names_model_colour[int(box.cls)]} {float(box.conf):.3}')


                    cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)
                    #int(center[0]*0.6666666666666667), int(center[1]*0.6666666666666667),
                    dist=compare_depth(depth_scale,int(center[0]), int(center[1]),calib_depth,depth_image,depth_image_raw)
                    #### SAVE in brick or create new 
                    for i in range(0,len(avg_center_list_x[:])):

                        if center[0]+50 > avg_center_list_x[i] >center[0]-50 and center[1]+50 > avg_center_list_y[i] >center[1]-50:
                            brick_index=i
                            #print('old Brick COLOUR')
                            New_brick=False
                            
                            list_of_x[i].append(center[0])
                            list_of_y[i].append(center[1])
                            list_of_colour[i].append([colour_type])
                            list_of_depth[i].append(dist)

                    if New_brick==True:
                        #print('New Brick COLOUR')
                        list_of_angle.append([])
                        list_of_x.append([center[0]])
                        list_of_y.append([center[1]])
                        
                        #points=np.array(Cor)
                        #ind = np.lexsort((points[:,0],points[:,1]))
                        #Cor=Cor[ind]

                        std_x.append([0])
                        std_y.append([0])
                        std_angle.append([])
                        
                        list_of_type.append([])
                        list_of_colour.append([colour_type])

                        Cur_brick_type.append([])
                        Cur_brick_colour.append([colour_type])
                        avg_of_angle.append([])
                        avg_center_list_x.append([center[0]])
                        avg_center_list_y.append([center[1]])

                        list_of_depth.append([dist])
                        avg_of_depth.append([dist])

                        

        if masks_type is not None:
            masks_type = masks_type.data.cpu()
            for seg, box in zip(masks_type.data.cpu().numpy(), boxes_type):
                
                seg = cv2.resize(seg, (w, h))
                img,Angle_temp = overlay(img, seg, colors[int(box.cls)], 0.4)
                if Angle_temp!=None:
                        
                    xmin = int(box.data[0][0])
                    ymin = int(box.data[0][1])
                    xmax = int(box.data[0][2])
                    ymax = int(box.data[0][3])
                    
                    
                    New_brick=True

                    center=np.zeros(2)
                    brick_type = 1                   
                    #print('LEGO is found')

                    #print(Cor)
                    center[0]=(xmax-xmin)/2+xmin
                    center[1]=(ymax-ymin)/2+ymin

                    #print(Cor)
                    if float(box.conf)>=0.3:
                        plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names_type[int(box.cls)]} {float(box.conf):.3}')

                        cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)

                        Brick_type=box.cls

                        #print(Brick_type)
                        
                        #### SAVE in brick or create new 
                        for i in range(0,len(avg_center_list_x[:])):

                            if center[0]+50 > avg_center_list_x[i] >center[0]-50 and center[1]+50 > avg_center_list_y[i] >center[1]-50:
                                brick_index=i
                                #print('Found existing')
                                New_brick=False
                                
                                
                                list_of_angle[i].append(Angle_temp)
                                #list_of_x[i].append(center[0])
                                #list_of_y[i].append(center[1])
                                list_of_type[i].append([Brick_type])
                        
                            if New_brick==True:
                                print('\n')
                                #list_of_angle.append([Angle_temp])
                                #list_of_x.append([center[0]])
                                #list_of_y.append([center[1]])
                                
                                #points=np.array(Cor)
                                #ind = np.lexsort((points[:,0],points[:,1]))
                                #Cor=Cor[ind]

                                #std_x.append([0])
                                #std_y.append([0])
                                #std_angle.append([0])
                                
                                #list_of_type.append([Brick_type])
                                #list_of_colour.append([])

                                #Cur_brick_type.append([Brick_type])
                                #Cur_brick_colour.append([])
                                #avg_of_angle.append([Angle_temp])
                                #avg_center_list_x.append([center[0]])
                                #avg_center_list_y.append([center[1]])


                    
        color_hitlist=np.zeros(len(list_of_x))
        type_hitlist=np.zeros(len(list_of_angle))

                    
        for i in range(0,len(avg_center_list_x[:])):
            avg_center_list_x[i]=np.mean(list_of_x[i])
            avg_center_list_y[i]=np.mean(list_of_y[i])
            avg_of_angle[i]=np.mean(list_of_angle[i])

            #print(list_of_depth[i])
            avg_of_depth[i]=np.mean(list_of_depth[i])
            color_hitlist[i]=len(list_of_x[i])
            type_hitlist[i]=len(list_of_angle[i])

            std_x[i]=np.std(list_of_x[i])
            std_y[i]=np.std(list_of_y[i])
            std_angle[i]=np.std(list_of_angle[i])
            if len(list_of_type[i])>=3:
                Cur_brick_type[i]=int(most_frequent(list_of_type[i])[0])



            if len(list_of_colour[i])>=3:
                Cur_brick_colour[i]=int(most_frequent(list_of_colour[i])[0])


            #print('This is the type: '+str(Cur_brick_type))
            #print('This is the colour: '+str(Cur_brick_colour))

            std_list=[std_x,std_y,std_angle]

        if t>=frames:# and any(std_1 for in std for std in std_list):
            Found=True
            result_x=avg_center_list_x
            result_y=avg_center_list_y
            result_angle=avg_of_angle
            result_depth=avg_of_depth

            Found=True
            break

        #cv2.imshow('canny',cannyImg)
        # First we crop the sub-rect from the image
        
        sub_img = img[int(roi[-2]):int(roi[-1]), int(roi[0]):int(roi[1])]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

        res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

        # Putting the image back to its position
        img[int(roi[-2]):int(roi[-1]), int(roi[0]):int(roi[1])]= res
        cv2.imshow('out',img)
        if Found==True:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'): # q closes our webcam
            break
        #Counter checking the number of pictures taken
        t=t+1
        print(t)
    print('list of angles')
    print(len(list_of_angle))
    #print(result_x)
    #print(result_y)

    return result_x, result_y, result_angle,Cur_brick_colour,Cur_brick_type,std_x,std_y,std_angle, class_names_type, class_names_colour,result_depth,color_hitlist,type_hitlist
    #print('finished')
    #print(list_of_type)
    #print(list_of_x)

    #print(list_of_y)
    #print(list_of_angle)