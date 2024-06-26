from ultralytics import YOLO
import cv2
import random
import numpy as np
from math import atan2, cos, sin, sqrt, pi
import pandas as pd
#import RV_Math 

def most_frequent(List):
    return max(List, key = List.count)

def drawAxis(img, p_, q_, colour, scale):
    p = list(p_)
    q = list(q_)
    
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
    return img

def getOrientation(pts, img):
 
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))


    cv2.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    img=drawAxis(img, cntr, p1, (0, 255, 0), 10)
    img=drawAxis(img, cntr, p2, (255, 255, 0), 30)
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return img,angle

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    img_empty = np.zeros((image.shape[0],image.shape[1],3), dtype=np.uint8)
    masked2 = np.ma.MaskedArray(img_empty, mask=colored_mask, fill_value=[255,255,255])
    masked2=masked2.filled()
    gray = cv2.cvtColor(masked2, cv2.COLOR_BGR2GRAY)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(10,10)))
    cv2.imshow('cam', gray)
    # Convert image to binary
    _, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours,_= cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area < 1e2 or 1e5 < area:
            continue
        # Draw each contour only for visualisation purposes
        #cv.drawContours(src, contours, i, (0, 0, 255), 2)
        
        # Find the orientation of each shape
        image_combined,angle=getOrientation(c, image_combined)
        #print("The angle is: "+str(angle))

    return image_combined, angle

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_xyA(frames):
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

    data_stream_of_x=[]
    data_stream_of_y=[]
    data_stream_of_angle=[]
    data_stream_of_type=[]
    data_stream_of_colour=[]

    std_x=[]
    std_y=[]
    std_angle=[]
    Found=False
    t=0
    Data_log=pd.DataFrame()
    t_list=[]

    print('Package Imported')


    # Load a model
    model_type = YOLO("Yolo-LEGO-test/runs/segment/train3/weights/best.pt")
    model_colour = YOLO("Yolo-LEGO-test/runs/detect/train10_farve_100_epocher/weights/best.pt")
    class_names_type = model_type.names
    print('Class Names: ', class_names_type)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names_type]
    # cap = cv2.VideoCapture('test.mp4')

    class_names_colour = model_colour.names
    print('Class Names: ', class_names_colour)

    webcam = cv2.VideoCapture(1)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -5)


    while True:

        succes, img = webcam.read() #define a variable called img, which is my webcam # success is a boolen which tells if we captured the video

        if t==0:
            cv2.imwrite('Machine-Vision/Log/img.png', img) 

        if not succes:
            break

        h, w, _ = img.shape
        #results = model.predict(img, stream=True)
        results_type = model_type.track(img, stream=True)
        results_colour = model_colour(img, stream=True)
        for r in results_type:
            boxes_type = r.boxes  # Boxes object for bbox outputs
            masks_type = r.masks  # Masks object for segment masks outputs
            probs_type = r.probs  # Class probabilities for classification outputs

        for r in results_colour:
            boxes_colour = r.boxes  # Boxes object for bbox outputs
            # Masks object for segment masks outputs
            probs_colour = r.probs  # Class probabilities for classification outputs

        if masks_type is not None:
            masks_type = masks_type.data.cpu()
            for seg, box in zip(masks_type.data.cpu().numpy(), boxes_type):
                
                seg = cv2.resize(seg, (w, h))
                img,Angle_temp = overlay(img, seg, colors[int(box.cls)], 0.4)
                
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

                plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names_type[int(box.cls)]} {float(box.conf):.3}')

                cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)

                Brick_type=box.cls

                #print(Brick_type)
                
                #### SAVE in brick or create new 
                for i in range(0,len(avg_center_list_x[:])):

                    if center[0]+20 > avg_center_list_x[i] >center[0]-20 and center[1]+20 > avg_center_list_y[i] >center[1]-20:
                        brick_index=i
                        New_brick=False
                        
                        data_stream_of_x.append([i,center[0],t])
                        data_stream_of_y.append([i,center[1],t])
                        data_stream_of_angle.append([i,Angle_temp,t])
                        data_stream_of_type.append([i,Brick_type,t])


                        
                        list_of_angle[i].append(Angle_temp)
                        list_of_x[i].append(center[0])
                        list_of_y[i].append(center[1])
                        list_of_type[i].append([Brick_type])
                
                if New_brick==True:
                    list_of_angle.append([Angle_temp])
                    list_of_x.append([center[0]])
                    list_of_y.append([center[1]])
                    
                    #points=np.array(Cor)
                    #ind = np.lexsort((points[:,0],points[:,1]))
                    #Cor=Cor[ind]

                    std_x.append([0])
                    std_y.append([0])
                    std_angle.append([0])
                    
                    list_of_type.append([Brick_type])
                    list_of_colour.append([])

                    Cur_brick_type.append([Brick_type])
                    Cur_brick_colour.append([])
                    avg_of_angle.append([Angle_temp])
                    avg_center_list_x.append([center[0]])
                    avg_center_list_y.append([center[1]])

        ####SKAL LIGE FIKSES
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
                
                #### SAVE in brick or create new 
                for i in range(0,len(avg_center_list_x[:])):

                    if center[0]+20 > avg_center_list_x[i] >center[0]-20 and center[1]+20 > avg_center_list_y[i] >center[1]-20:
                        brick_index=i
                        New_brick=False

                        data_stream_of_x.append([i,center[0],t])
                        data_stream_of_y.append([i,center[1],t])
                        data_stream_of_angle.append([i,Angle_temp,t])
                        data_stream_of_colour.append([i,colour_type,t])
                        
                        list_of_angle[i].append(Angle_temp)
                        list_of_x[i].append(center[0])
                        list_of_y[i].append(center[1])
                        list_of_colour[i].append([colour_type])


                    
        for i in range(0,len(avg_center_list_x[:])):
            avg_center_list_x[i]=np.mean(list_of_x[i])
            avg_center_list_y[i]=np.mean(list_of_y[i])
            avg_of_angle[i]=np.mean(list_of_angle[i])

            std_x[i]=np.std(list_of_x[i])
            std_y[i]=np.std(list_of_y[i])
            std_angle[i]=np.std(list_of_angle[i])
            if len(list_of_type[i])>=1:
                Cur_brick_type[i]=int(most_frequent(list_of_type[i])[0])

            if len(list_of_colour[i])>=1:
                Cur_brick_colour[i]=int(most_frequent(list_of_colour[i])[0])

            #print('This is the type: '+str(Cur_brick_type))
            #print('This is the colour: '+str(Cur_brick_colour))

            #std_list=[std_x,std_y,std_angle]
        t_list=[]
        for k in range(0,len(Cur_brick_colour)):
            t_list.append(t)
        #Data_array=np.append(Data_array,[[avg_center_list_x], [avg_center_list_y], [avg_of_angle],[Cur_brick_colour],[Cur_brick_type],[std_x],[std_y],[std_angle], [class_names_type], [class_names_colour], [t_list]])
        
        Data_df=pd.DataFrame({'x-Cordinates [px]' : avg_center_list_x,'y-Cordinates [px]' : avg_center_list_y, 'Angle [rad]' : avg_of_angle,'Brick Colour' : Cur_brick_colour,'Brick Type':Cur_brick_type, 'Standard Deviation x' : std_x,'Standard Deviation y' : std_y,'Standard Deviation Angle' : std_angle,'Frame' : t_list})
    
        Data_log=pd.concat([Data_log, Data_df])
        if t>=frames:# and any(std_1 for in std for std in std_list):
            Found=True
            result_x=avg_center_list_x
            result_y=avg_center_list_y
            result_angle=avg_of_angle



            Found=True
            break



        #cv2.imshow('canny',cannyImg)
        cv2.imshow('out',img)
        if Found==True:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'): # q closes our webcam
            break
        #Counter checking the number of pictures taken
        t=t+1

    return result_x, result_y, result_angle,Cur_brick_colour,Cur_brick_type,std_x,std_y,std_angle, class_names_type, class_names_colour,Data_log, data_stream_of_x,data_stream_of_y,data_stream_of_angle,data_stream_of_type,data_stream_of_colour
    #print('finished')
    #print(list_of_type)
    #print(list_of_x)
    #print(list_of_y)
    #print(list_of_angle)

if __name__ == '__main__':
    result_x, result_y, result_angle,Cur_brick_colour,Cur_brick_type,std_x,std_y,std_angle, class_names_type, class_names_colour,Data_log, list_of_x,list_of_y,list_of_angle,list_of_type,list_of_colour =get_xyA(100)

    results=pd.DataFrame({'x-Cordinates [px]' : result_x,'y-Cordinates [px]' : result_y, 'Angle [rad]' : result_angle,'Brick Colour' : Cur_brick_colour,'Brick Type':Cur_brick_type, 'Standard Deviation x' : std_x,'Standard Deviation y' : std_y,'Standard Deviation Angle' : std_angle})
    results=results.replace({"Brick Colour": class_names_colour,"Brick Type": class_names_type})
    print(class_names_type)
    print(class_names_colour)
    
    #Data_results=pd.DataFrame({'x-Cordinates [px]' : Data_array[:,0],'y-Cordinates [px]' : Data_array[:,1], 'Angle [rad]' : Data_array[:,2],'Brick Colour' : Data_array[:,3],'Brick Type':Data_array[:,4], 'Standard Deviation x' : Data_array[:,5],'Standard Deviation y' : Data_array[:,6],'Standard Deviation Angle' : Data_array[:,7], 'Frame' : Data_array[:,8]})
    Data_log=Data_log.replace({"Brick Colour": class_names_colour,"Brick Type": class_names_type})

    print(Data_log)

    #SAVE THE FOllOWING
    #Data_log, list_of_x,list_of_y,list_of_angle,list_of_type,list_of_colour
    list_of_x = pd.DataFrame(list_of_x)
    list_of_y = pd.DataFrame(list_of_y)
    list_of_angle = pd.DataFrame(list_of_angle)
    list_of_type = pd.DataFrame(list_of_type)
    list_of_colour = pd.DataFrame(list_of_colour)


    #np.savetxt("LEGO-Master/Machine-Vision/Log/list_of_x.csv", list_of_x, delimiter=",")
    #np.savetxt("LEGO-Master/Machine-Vision/Log/list_of_y.csv", list_of_y, delimiter=",")
    #np.savetxt("LEGO-Master/Machine-Vision/Log/list_of_angle.csv", list_of_angle, delimiter=",")
    #np.savetxt("LEGO-Master/Machine-Vision/Log/list_of_type.csv", list_of_type, delimiter=",")
    #np.savetxt("LEGO-Master/Machine-Vision/Log/list_of_colour.csv", list_of_colour, delimiter=",")

    pd.DataFrame(list_of_x).to_csv("Machine-Vision/Log/list_of_x.csv")
    pd.DataFrame(list_of_y).to_csv("Machine-Vision/Log/list_of_y.csv")
    pd.DataFrame(list_of_angle).to_csv("Machine-Vision/Log/list_of_angle.csv")
    pd.DataFrame(list_of_type).to_csv("Machine-Vision/Log/list_of_type.csv")
    pd.DataFrame(list_of_colour).to_csv("Machine-Vision/Log/list_of_colour.csv")


    pd.DataFrame(Data_log).to_csv('Machine-Vision/Log/Results_Log.csv') 
      







