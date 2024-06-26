import os
import numpy as np
import cv2


input_dir = 'Yolo-LEGO-test/physical_data/data/SegmentationObject'
class_path='Yolo-LEGO-test/physical_data/data/SegmentationClass'
class_dir=os.listdir('Yolo-LEGO-test/physical_data/data/SegmentationClass')
output_dir = 'Yolo-LEGO-test/physical_data/data/Labels/'
class_dict={255:0,83:1,55:3,124:4}
for j in os.listdir(input_dir):
    print('image:' +j)
    image = cv2.imread(os.path.join(input_dir, j))
    # load the binary mask and get its contours

    # Convert image to RGB color space
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels
    pixels = image_rgb.reshape(-1, 3)

    # Get unique colors
    unique_colors = np.unique(pixels, axis=0)

    print(unique_colors[1])
    polygons_list=[]
    classes = []
    for i in range(0,len(unique_colors)):
        # Define the target color (in RGB)
        target_color = np.array(unique_colors[i])  # Replace with your target color values
        target_color=[target_color[-1],target_color[-2],target_color[-3]]
        #print('Target colour:'+ str(target_color))
        # Create a mask to filter the image based on the exact match with the target color
        mask = np.all(image == target_color, axis=2)

        # Apply the mask to the original image to get the filtered image
        filtered_image = np.zeros_like(image)
        filtered_image[mask] = image[mask]
        

        mask = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY) 
        #_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        #cv2.imshow('Filtered Image', mask)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        H, W = mask.shape
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # convert the contours to polygons
        polygons = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 2:
                polygon = []
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    class_image = cv2.imread(os.path.join(class_path, j))
                    #print('Colour:'+ str(class_image[cy, cx]))
                    #print(j)
                    if class_image[cy, cx][0]!=0:
                        print('colour: '+str(class_image[cy, cx]))
                        print('Class: ' +str(class_dict[class_image[cy, cx][0]]))
                        classes.append(class_dict[class_image[cy, cx][0]])
                        for point in cnt:
                            x, y = point[0]
                            polygon.append(x / W)
                            polygon.append(y / H)
                        polygons.append(polygon)
                        polygons_list.append(polygons)
            # print the polygons
    #print(polygons_list)
    #print(len(polygons_list))
    #print(len(classes))
    with open('{}.txt'.format(str(os.path.join(output_dir, j)[:-4])), 'w') as f:
        for t in range(0,len(polygons_list)):
            polytemp=polygons_list[t]
            for polygon in polytemp:
                #print(polygon)
                for p_, p in enumerate(polygon):
                    if p_ == len(polygon) - 1:
                        #print('End of polygon')
                        f.write('{}\n'.format(p))

                    elif p_ == 0:
                        f.write(str(classes[t])+' {} '.format(p))
                    else:
                        f.write('{} '.format(p))
                    

        f.close()