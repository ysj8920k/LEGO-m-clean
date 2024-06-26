import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Yolo-LEGO-test\physical_data\data\SegmentationClass/43.png')
image = cv2.imread('Yolo-LEGO-test\physical_data\data\SegmentationClass/1715241117.png')

# Convert image to RGB color space
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels
pixels = image_rgb.reshape(-1, 3)

# Get unique colors
unique_colors = np.unique(pixels, axis=0)

# Create a blank canvas to show colors
color_palette = np.zeros((100, 100 * len(unique_colors), 3), dtype=np.uint8)

# Display each unique color in the palette
for i, color in enumerate(unique_colors):
    color_palette[:, i * 100:(i + 1) * 100] = color

# Show the color palette
plt.imshow(color_palette)
plt.axis('off')
plt.show()
