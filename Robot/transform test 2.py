import numpy as np

def calculate_transformation_matrix(pixel_coords, mm_coords):
    # Extract pixel and real-world coordinates
    pixel_x, pixel_y, pixel_rot = pixel_coords
    mm_x, mm_y, mm_rot = mm_coords

    # Calculate translation components
    tx = mm_x - (pixel_x * np.cos(pixel_rot) + pixel_y * np.sin(pixel_rot))
    ty = mm_y - (-pixel_x * np.sin(pixel_rot) + pixel_y * np.cos(pixel_rot))
    
    # Calculate rotation component (assuming no rotation for simplicity)
    theta = -pixel_rot  # Inverse rotation angle to align with the coordinate system

    # Derive transformation matrix
    transformation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])

    return transformation_matrix



def calculate_pixel_to_mm_ratio(pixel_coords, mm_coords):
    # Calculate pixel-to-mm ratio for x and y axes separately
    pixel_to_mm_x = (mm_coords[0] - mm_coords[1]) / (pixel_coords[0] - pixel_coords[1])
    pixel_to_mm_y = (mm_coords[1] - mm_coords[2]) / (pixel_coords[1] - pixel_coords[2])

    # Average the ratios for a more stable estimate
    pixel_to_mm_avg = abs((pixel_to_mm_x + pixel_to_mm_y) / 2.0)

    return pixel_to_mm_avg


# Given pixel and real-world coordinates
pixel_coords = [
    [425.047, 421.020, 0.1725],
    [325.5, 73.5, 0.1782]
]
mm_coords = [
    [-5.914, 193.496, 0],
    [48.195, 0.240, 0]
]

# Calculate transformation matrix
T_matrix = calculate_transformation_matrix(pixel_coords[0], mm_coords[0])

# Calculate pixel-to-mm ratio
px_pr_mm = calculate_pixel_to_mm_ratio(pixel_coords[0], mm_coords[0])

print("Transformation Matrix:")
print(T_matrix)
print("Pixel-to-mm Ratio:", px_pr_mm)

def apply_transformation(pixel_coords, T_matrix, px_pr_mm):
    # Extract transformation parameters
    cos_theta = T_matrix[0][0]
    sin_theta = T_matrix[1][0]
    tx = T_matrix[0][2]
    ty = T_matrix[1][2]

    # Apply transformation
    x_px = pixel_coords[0]
    y_px = pixel_coords[1]
    theta = pixel_coords[2]

    x_mm = (cos_theta * x_px - sin_theta * y_px + tx) * px_pr_mm
    y_mm = (sin_theta * x_px + cos_theta * y_px + ty) * px_pr_mm
    theta_rad = theta  # No conversion needed for rotation angle

    return [x_mm, y_mm, theta_rad]

# Test transformation on both sets of pixel coordinates
pixel_coords_1 = [425.047, 421.020, 0.1725]
pixel_coords_2 = [325.5, 73.5, 0.1782]

transformed_coords_1 = apply_transformation(pixel_coords_1, T_matrix, px_pr_mm)
transformed_coords_2 = apply_transformation(pixel_coords_2, T_matrix, px_pr_mm)

# Print calculated transformed coordinates
print("Calculated transformed coordinates for pixel_coords_1:", transformed_coords_1)
print("Calculated transformed coordinates for pixel_coords_2:", transformed_coords_2)

# Expected transformed coordinates
expected_transformed_coords_1 = [-5.914, 193.496, 0]
expected_transformed_coords_2 = [48.195, 0.240, 0]

# Check if transformed coordinates match expected results
if np.allclose(transformed_coords_1, expected_transformed_coords_1, atol=1e-3) and \
   np.allclose(transformed_coords_2, expected_transformed_coords_2, atol=1e-3):
    print("Transformation check passed!")
else:
    print("Transformation check failed!")
