import cv2
import numpy as np

# Load the original image
image = cv2.imread('Yolo-LEGO-test\physical_data\data\SegmentationObject/1.png')

# Convert image to RGB color space
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image_rgb.reshape(-1, 3)

# Get unique colors
unique_colors = np.unique(pixels, axis=0)

print(unique_colors[1])
# Define the target color (in RGB)
target_color = np.array(unique_colors[1])  # Replace with your target color values
# Create a mask to filter the image based on the exact match with the target color
mask = np.all(image == target_color, axis=2)

# Apply the mask to the original image to get the filtered image
filtered_image = np.zeros_like(image)
filtered_image[mask] = image[mask]

# Display the filtered image
cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


mask = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY) 
_, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

H, W = mask.shape
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#class_dict={[51,221,255]:0,[250,50,83]:1,[250,250,55]:3}
class_dict={255:0,83:1,55:3}
# convert the contours to polygons
polygons = []
for cnt in contours:
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        class_image = cv2.imread('Yolo-LEGO-test\physical_data\data\SegmentationClass/1.png')
        cv2.circle(filtered_image, (cx, cy), 7, (0, 0, 255), -1)
        print(class_image[cy, cx])
        print('Class: ' +str(class_dict[class_image[cy, cx][0]]))

    


cv2.imshow('Filtered Image', filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
