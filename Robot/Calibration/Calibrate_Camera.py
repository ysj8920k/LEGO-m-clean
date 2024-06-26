from ultralytics import YOLO
import cv2
import random
import numpy as np
from math import atan2, cos, sin, sqrt, pi, radians
import math
import pandas as pd
import pyrealsense2 as rs
import json
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
    # Convert angle to degrees
    angle_deg = math.degrees(angle_rad)
    
    angle_deg -= 90
    # Normalize angle to be within the range [-90, 90)
    while angle_deg >= 90:
        angle_deg -= 180
    while angle_deg < -90:
        angle_deg += 180
    
    # Convert angle back to radians
    angle_rad = math.radians(angle_deg)
    
    return angle_rad


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

    #cv2.imshow('cam', gray)
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
        H_W_ratio = rect[1][0] / rect[1][1]
        if H_W_ratio < 0.85:
            angle = (rect[2]*np.pi)/180
        elif H_W_ratio > 1.15:
            angle = ((rect[2]+90)*np.pi)/180
        else:
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
    model_type = YOLO("Yolo-LEGO-test/runs/segment/train_Medium/weights/best.pt")
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
                        dist=compare_depth(depth_scale,int(center[0]), int(center[1]),calib_depth,depth_image,depth_image_raw)                        
                        #### SAVE in brick or create new 
                        for i in range(0,len(avg_center_list_x[:])):

                            if center[0]+50 > avg_center_list_x[i] >center[0]-50 and center[1]+50 > avg_center_list_y[i] >center[1]-50:
                                brick_index=i
                                print('Found existing')
                                New_brick=False
                                
                                
                                list_of_angle[i].append(Angle_temp)
                                list_of_x[i].append(center[0])
                                list_of_y[i].append(center[1])

                                list_of_depth[i].append(dist)
                                list_of_type[i].append([Brick_type])
                        
                        if New_brick==True:
                            print('\n')
                            print('Found New Type')
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

                            list_of_depth.append([dist])
                            avg_of_depth.append([dist])

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

                        if float(box.conf)>=0.4:
                            plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names_model_colour[int(box.cls)]} {float(box.conf):.3}')


                            cv2.circle(img, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)
                            #int(center[0]*0.6666666666666667), int(center[1]*0.6666666666666667),

                            #### SAVE in brick or create new 
                            for i in range(0,len(avg_center_list_x[:])):

                                if center[0]+50 > avg_center_list_x[i] >center[0]-50 and center[1]+50 > avg_center_list_y[i] >center[1]-50:
                                    brick_index=i
                                    print('old Brick COLOUR')
                                    New_brick=False
                                    
                                    #list_of_x[i].append(center[0])
                                    #list_of_y[i].append(center[1])
                                    list_of_colour[i].append([colour_type])




                        

                    
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

def initialize():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    #config.enable_stream(rs.stream.depth)
    #config.load_settings_json
    #if device_product_line == 'L500':
    #    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    #else:
    #    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    # Start streaming
    profile=pipeline.start(config)
    #SET EXPOSORE
    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 150.000)


    depth_sensor = profile.get_device().first_depth_sensor()
    #if depth_sensor.supports(rs.option.depth_units):
    #    depth_sensor.set_option(rs.option.depth_units, 0.0001)
    laser=False
    #if depth_sensor.supports(rs.option.emitter_enabled):
    #    depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if laser else 0.0)
    #elif laser:
    #    raise EnvironmentError('Device does not support laser')
    return pipeline,profile

def depth_calibration_raw(pipeline,number_of_frames):
    # Read the CSV file into a DataFrame
    # Load mtx from CSV
    mtx = np.genfromtxt('camera_matrix.csv', delimiter=',')

    # Load dist from CSV
    dist = np.genfromtxt('distortion_coefficients.csv', delimiter=',')
    depth_image_list=[]
    depth_colour_list=[]
    t=0
    while t<=number_of_frames:
        frameset = pipeline.wait_for_frames()

        depth_image,depth_image_raw ,color_image=post_process(frameset)

        #depth_image = undistort_image(depth_image, mtx, dist)
        #depth_image_raw = undistort_image(depth_image_raw, mtx, dist)
        depth_image_list.append(depth_image_raw)
        depth_colour_list.append(depth_image)
        t+=1

    calib_depth=avg_depth_depth(depth_image_list)
    avg_depth_colour(depth_colour_list)
    return calib_depth

def avg_depth_depth(depth_images):


    image_data=depth_images
        # Calculate blended image

    dst=np.mean( image_data, axis=0 )
            
    cv2.imwrite('Robot\Calibration/average_depth_gray.png',dst)
    return dst

def avg_depth_colour(depth_images):


    image_data=depth_images
    # Calculate blended image
    # Calculate blended image
    dst = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(image_data[i], alpha, dst, beta,0.0)
            

            
    cv2.imwrite('Robot\Calibration/average_depth_colour.png',dst)
    return dst

def post_process(frameset):
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    # Update color and depth frames:
    
    aligned_depth_frame = frameset.get_depth_frame()
    colorizer = rs.colorizer()

    #depth_frame = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    #depth_frame = frames.get_depth_frame()
    color_frame = frameset.get_color_frame()
    #if not depth_frame or not color_frame:
    #    continue
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)


    temporal = rs.temporal_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)
    aligned_depth_frame = decimation.process(aligned_depth_frame)
    aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
    aligned_depth_frame = spatial.process(aligned_depth_frame)
    aligned_depth_frame = temporal.process(aligned_depth_frame)
    aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
    
    # Convert images to numpy arrays
    depth_image =  np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
    return depth_image,depth_image_raw,color_image

def compare_depth(depth_scale,x,y,calib_raw,depth_img,depth_raw):
    image=depth_img
    image = cv2.circle(image, (x,y), radius=10, color=(255, 255, 255), thickness=-1)

    # Crop depth data:
    #depth = depth[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)

    # Get data scale from the device and convert to meters
    #depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    #depth = depth * depth_scale
    #dist,_,_,_ = cv2.mean(depth)

    #img_val = depth_raw[x-10:x+10, y-10:y+10].astype(float)
    #cal_val = calib_raw[x-10:x+10, y-10:y+10].astype(float)
    img_val = depth_raw[y-10:y+10,x-10:x+10].astype(float)
    cal_val = calib_raw[y-10:y+10,x-10:x+10].astype(float)
    img_val=img_val*depth_scale
    cal_val=cal_val*depth_scale
    #dist_img,_,_,_ = cv2.mean(img_val)
    #dist_cal,_,_,_ = cv2.mean(cal_val)
    dist_img = np.mean(img_val)
    dist_cal = np.mean(cal_val)

    #print('Current')
    #print(dist_img)

    #print('Calib')
    #print(dist_cal)

    #print('diff')
    #print((dist_cal-dist_img))
    cv2.imshow('depth', image)
    cv2.waitKey(1)
    return (dist_cal-dist_img)*1000

#Rworld_cords=np.array([-3.958793, -67.642961,  -0.019019, 190.146908]

def main_mv_belt(min_frames, webcam,profile, calib_depth, Rworld_cords=np.array([0, -135.885,  0, 194.822]), roi=[0, 100, 0, 100],T_inpx=[0,0,0], px_pr_mm = 0, Calibrating=True):
    if Calibrating:
        result_x, result_y, result_angle, Cur_brick_colour, Cur_brick_type, std_x, std_y, std_angle, class_names_type, class_names_colour,depth,color_hits,type_hits = get_xyA(
            100, webcam, roi,profile, calib_depth)

        results = pd.DataFrame({'x-Cordinates [px]': result_x, 'y-Cordinates [px]': result_y,
                                'Angle [rad]': result_angle, 'Brick Colour': Cur_brick_colour,
                                'Brick Type': Cur_brick_type, 'Standard Deviation x': std_x,
                                'Standard Deviation y': std_y, 'Standard Deviation Angle': std_angle,
                                'Depth':depth,'Color Model Hits':color_hits,'Type Model Hits':type_hits})
        results = results.replace(
            {"Brick Colour": class_names_colour, "Brick Type": class_names_type})

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
            roi = [min(results['x-Cordinates [px]']) + (0 * px_pr_mm), 
                max(results['x-Cordinates [px]']) + (30 * px_pr_mm), 
                min(results['y-Cordinates [px]']) + (25 * px_pr_mm), 
                max(results['y-Cordinates [px]']) - (40 * px_pr_mm)]
            #roi = [0,1920,0,1080]

            # Calculate the average angle for T_inpx
            avg_angle = sum(results['Angle [rad]']) / len(results['Angle [rad]'])
            T_inpx = [min(results['x-Cordinates [px]']), min(results['y-Cordinates [px]']), avg_angle]

            print('The region of interest is:  ' + str(roi))
            print('PX to mm ratio:  ' + str(px_pr_mm))
            print('Camera 0 is offset, by:  ' + str(T_inpx) + ' in relation to Robot Frame')
            brick_one=[-(min(results['x-Cordinates [px]'])-T_inpx[0]) / px_pr_mm, 
                (min(results['y-Cordinates [px]'])-T_inpx[1]) / px_pr_mm]

            brick_two=[-(max(results['x-Cordinates [px]'])-T_inpx[0]) / px_pr_mm, 
               (max(results['y-Cordinates [px]'])-T_inpx[1]) / px_pr_mm]

            with open("Robot/Calibration/Calibration_Data.json", 'w') as f:
                # indent=2 is not needed but makes the file human-readable 
                # if the data is nested
                json.dump([T_inpx,px_pr_mm,roi,brick_one,brick_two], f, indent=2) 


        else:
            print("More or less than two bricks were found")
            return
        


 




if __name__ == '__main__':
    pipeline,profile=initialize()
    calib_depth=depth_calibration_raw(pipeline,100)
    #to_continue()
    main_mv_belt(100,pipeline,profile,calib_depth,Calibrating=True)
