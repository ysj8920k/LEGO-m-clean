from ultralytics import YOLO
import cv2
import random
import numpy as np
from math import atan2, cos, sin, sqrt, pi
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

def get_xyA(type,colour):
    avg_center_list_x=[]
    avg_center_list_y=[]
    avg_of_angle=[]
    list_of_x=[]
    list_of_y=[]
    list_of_angle=[]
    list_of_type=[]
    list_of_colour=[]
    
    Found=False
    t=0
    print('Package Imported')


    # Load a model
    model_type = YOLO("Yolo-LEGO-test/runs/segment/train4/weights/best.pt")
    model_colour = YOLO("Yolo-LEGO-test/runs/detect/train10_farve_100_epocher/weights/best.pt")
    class_names_type = model_type.names
    print('Class Names: ', class_names_type)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names_type]
    # cap = cv2.VideoCapture('test.mp4')

    class_names_colour = model_colour.names
    print('Class Names: ', class_names_colour)

    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_EXPOSURE, -5)


    while True:

        succes, img = webcam.read() #define a variable called img, which is my webcam # success is a boolen which tells if we captured the video


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


                    
                    list_of_type.append([Brick_type])
                    list_of_colour.append([])

                    
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
                        
                        list_of_angle[i].append(Angle_temp)
                        list_of_x[i].append(center[0])
                        list_of_y[i].append(center[1])
                        list_of_colour[i].append([colour_type])


                    
        for i in range(0,len(avg_center_list_x[:])):
            avg_center_list_x[i]=np.mean(list_of_x[i])
            avg_center_list_y[i]=np.mean(list_of_y[i])
            avg_of_angle[i]=np.mean(list_of_angle[i])

            std_x=np.std(list_of_x[i])
            std_y=np.std(list_of_y[i])
            std_angle=np.std(list_of_angle[i])
            if len(list_of_type[i])>=3:
                Cur_brick_type=int(most_frequent(list_of_type[i])[0])
            else:
                Cur_brick_type='None'


            if len(list_of_colour[i])>=3:
                Cur_brick_colour=int(most_frequent(list_of_colour[i])[0])
            else:
                Cur_brick_colour='None'

            print('This is the type: '+str(Cur_brick_type))
            print('This is the colour: '+str(Cur_brick_colour))

            std_list=[std_x,std_y,std_angle]
            if all(std<1 for std in std_list) and t>=20 and Cur_brick_type==type  and Cur_brick_colour == colour and len(list_of_x[i])>20:
                
                
                print('Succes: The brick chosen was:' + str(i)+'    Which is type: '+str(Cur_brick_type) +'    And colour: '+str(Cur_brick_colour))
                print('Which had the following deviations: X:'+str(std_x)+'  Y:'+str(std_y)+'  Angle: '+str(std_angle))
                print('Which had the following location: X:'+str( avg_center_list_x[i])+'  Y:'+str( avg_center_list_y[i])+'  Angle:'+str(avg_of_angle[i]))
                print('And took ' +str(t)+' Frames to get')
                Found=True
                result_x=avg_center_list_x[i]
                result_y=avg_center_list_y[i]
                result_angle=avg_of_angle[i]
                
                break
            elif t>=2000:
            
                print("No brick was found")
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
    return result_x, result_y, result_angle
    #print('finished')
    #print(list_of_type)
    #print(list_of_x)
    #print(list_of_y)
    #print(list_of_angle)

if __name__ == '__main__':
    get_xyA(0,0)
      







