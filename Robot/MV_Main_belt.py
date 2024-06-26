from ultralytics import YOLO
import cv2
import random
import numpy as np
from math import atan2, cos, sin, sqrt, pi, radians
import math
import pandas as pd
#import RV_Math 
def calc_length(x,y):
    l1 = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    return l1

def to_continue():
    while True:
        choice = input("is it calibrated press b to continue")
        if choice == 'b' :

            break

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]

    # Directly pass mtx as a NumPy array
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
    return undistorted_img

def most_frequent(List):
    return max(List, key = List.count)

def normalize_angle(angle_rad):
    # Ensure the angle is within the range [0, 2*pi)
    angle_rad %= 2 * math.pi
    
    # Adjust the angle to be within the range [0, pi/2)
    while angle_rad >= math.pi / 2:
        angle_rad -= math.pi / 2
    
    return angle_rad

def is_valid_brick(row):
    return row['Brick Type'] in class_names_type and row['Brick Colour'] in class_names_colour


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

    # Normalize the angle
    angle = normalize_angle(angle)

    return img,angle

def overlay(image, mask, color, alpha, resize=None):
    angle=None
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
    #gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))) #CLOSING

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
        # Find the minimum bounding rectangle
        rect = cv2.minAreaRect(c)
        angle = (rect[2]*np.pi)/180
        
        # Normalize the angle
        angle = normalize_angle(angle)
     

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




def get_xyA(frames,webcam,roi):
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

        succes, img = webcam.read() #define a variable called img, which is my webcam # success is a boolen which tells if we captured the video
        # Read the CSV file into a DataFrame
      # Load mtx from CSV
        mtx = np.genfromtxt('camera_matrix.csv', delimiter=',')

        # Load dist from CSV
        dist = np.genfromtxt('distortion_coefficients.csv', delimiter=',')




        img = undistort_image(img, mtx, dist)



        if not succes:
            break

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

                if float(box.conf)>=0.8:
                    plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names_model_colour[int(box.cls)]} {float(box.conf):.3}')


                    cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)
                
                    #### SAVE in brick or create new 
                    for i in range(0,len(avg_center_list_x[:])):

                        if center[0]+50 > avg_center_list_x[i] >center[0]-50 and center[1]+50 > avg_center_list_y[i] >center[1]-50:
                            brick_index=i
                            #print('old Brick COLOUR')
                            New_brick=False
                            
                            list_of_x[i].append(center[0])
                            list_of_y[i].append(center[1])
                            list_of_colour[i].append([colour_type])

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
                    if float(box.conf)>=0.8:
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
                                print('Found new brick type')
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


                    


                    
        for i in range(0,len(avg_center_list_x[:])):
            avg_center_list_x[i]=np.mean(list_of_x[i])
            avg_center_list_y[i]=np.mean(list_of_y[i])
            avg_of_angle[i]=np.mean(list_of_angle[i])

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

    return result_x, result_y, result_angle,Cur_brick_colour,Cur_brick_type,std_x,std_y,std_angle, class_names_type, class_names_colour
    #print('finished')
    #print(list_of_type)
    #print(list_of_x)

    #print(list_of_y)
    #print(list_of_angle)


import numpy as np
import pandas as pd

def main_mv_belt(min_frames, webcam, Rworld_cords=np.array([0, -56, 0, 192]), roi=[0, 100, 0, 100],T_inpx=[0,0,0], px_pr_mm = 0, Calibrating=True):
    if Calibrating:
        result_x, result_y, result_angle, Cur_brick_colour, Cur_brick_type, std_x, std_y, std_angle, class_names_type, class_names_colour = get_xyA(
            100, webcam, roi)

        results = pd.DataFrame({'x-Cordinates [px]': result_x, 'y-Cordinates [px]': result_y,
                                'Angle [rad]': result_angle, 'Brick Colour': Cur_brick_colour,
                                'Brick Type': Cur_brick_type, 'Standard Deviation x': std_x,
                                'Standard Deviation y': std_y, 'Standard Deviation Angle': std_angle})
        results = results.replace(
            {"Brick Colour": class_names_colour, "Brick Type": class_names_type})
        
        #print('unfiltered results')
        #print(results)
        print(class_names_type)
        print(class_names_colour)

        for index, row in results.iterrows():
            if row['Brick Type'] not in class_names_type.values() or row['Brick Colour'] not in class_names_colour.values():
                print(f"Row with color '{row['Brick Colour']}' and type '{row['Brick Type']}' was dropped")
                results.drop(index, inplace=True)

        print("Filtered results:")
        print(results)


        

        

        if len(results) == 2:
            # Calculate px_pr_mm, roi, and T_inpx only if there are exactly 2 valid bricks
            px_length = calc_length([min(results['x-Cordinates [px]']), max(results['x-Cordinates [px]'])], 
                                    [min(results['y-Cordinates [px]']), max(results['y-Cordinates [px]'])])
            mm_length = calc_length([Rworld_cords[0], Rworld_cords[1]], [Rworld_cords[2], Rworld_cords[3]])
            px_pr_mm = px_length / mm_length

            # Adjust roi based on px_pr_mm
            roi = [min(results['x-Cordinates [px]']) - (50 * px_pr_mm), 
                max(results['x-Cordinates [px]']) + (50 * px_pr_mm), 
                min(results['y-Cordinates [px]']) + (25 * px_pr_mm), 
                max(results['y-Cordinates [px]']) - (25 * px_pr_mm)]

            # Calculate the average angle for T_inpx
            avg_angle = sum(results['Angle [rad]']) / len(results['Angle [rad]'])
            T_inpx = [min(results['x-Cordinates [px]']), min(results['y-Cordinates [px]']), avg_angle]

            print('The region of interest is:  ' + str(roi))
            print('PX to mm ratio:  ' + str(px_pr_mm))
            print('Camera 0 is offset, by:  ' + str(T_inpx) + ' in relation to Robot Frame')
            to_continue()
        else:
            print("More or less than two bricks were found")
            return


    result_x, result_y, result_angle, Cur_brick_colour, Cur_brick_type, std_x, std_y, std_angle, class_names_type, class_names_colour = get_xyA(
        min_frames, webcam, roi)

    results = pd.DataFrame({'x-Cordinates [px]': result_x, 'y-Cordinates [px]': result_y,
                            'Angle [rad]': result_angle, 'Brick Colour': Cur_brick_colour,
                            'Brick Type': Cur_brick_type, 'Standard Deviation x': std_x,
                            'Standard Deviation y': std_y, 'Standard Deviation Angle': std_angle})
    results = results.replace(
        {"Brick Colour": class_names_colour, "Brick Type": class_names_type})

    results = results.loc[(roi[1] > results['x-Cordinates [px]']) & (results['x-Cordinates [px]'] > +roi[0]) & (
            roi[3] > results['y-Cordinates [px]']) & (results['y-Cordinates [px]'] > +roi[2])]
    
    for index, row in results.iterrows():
            if row['Brick Type'] not in class_names_type.values() or row['Brick Colour'] not in class_names_colour.values():
                print(f"Row with color '{row['Brick Colour']}' and type '{row['Brick Type']}' was dropped")
                results.drop(index, inplace=True)

    print("Filtered results:")
    print(results)


    print("The following was found")
    print(class_names_type)
    print(class_names_colour)
    print(roi)
    print(results)

    print("Transformation matrix")
    print(T_inpx)

   # Iterate over each row in the "results" DataFrame
    for index, row in results.iterrows():
        # Create a vector from the x and y coordinates of the current row
        vector = np.array([row['x-Cordinates [px]'], row['y-Cordinates [px]']])

        # Subtract the first value from T_inpx from the vector's x-coordinate
        # Subtract the second value from T_inpx from the vector's y-coordinate
        vector -= np.array(T_inpx[:2])

        # Rotate the vector by a number of radians equal to the negative of the third index in T_inpx
        theta = -T_inpx[2]
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],   # Notice the change here
                                    [np.sin(theta), np.cos(theta)]]) # Negate the sin component
        
        rotated_vector = np.dot(rotation_matrix, vector)

        # Convert the rotated vector components to millimeters using px_pr_mm
        vector_mm = rotated_vector / px_pr_mm

        # Update the "x-Cordinates [px]" and "y-Cordinates [px]" columns of the current row with the converted values
        results.at[index, 'x-Cordinates [px]'] = -(vector_mm[0]-4)
        results.at[index, 'y-Cordinates [px]'] = vector_mm[1]


    print("Updated results DataFrame:")
    print(results)

    results=results.values



    return results,roi,T_inpx,px_pr_mm




if __name__ == '__main__':
    webcam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -6 )
    main_mv_belt(100,webcam,Calibrating=True)
