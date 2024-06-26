import math

def normalize_angle(angle_rad):
    # Normalize angle to be within the range [-pi/2, pi/2]
    while angle_rad >= math.pi / 2:
        angle_rad -= math.pi
    while angle_rad < -math.pi / 2:
        angle_rad += math.pi
    
    return angle_rad

# Test the function with a single angle
angle_to_test = -5 * math.pi / 180  # Convert angle to radians
normalized_angle = normalize_angle(angle_to_test)
normalized_angle_deg = math.degrees(normalized_angle)

print("Original angle (degrees):", math.degrees(angle_to_test))
print("Normalized angle (degrees):", normalized_angle_deg)
