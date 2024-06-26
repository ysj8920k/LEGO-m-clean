import numpy as np

def calculate_offsets(new_point):
    # Define points and their offsets
    points_offsets = [
        (np.array([-24.891303, 33.454318]), np.array([4.89, -2.365])),
        (np.array([-93.181886, 97.232641]), np.array([6.113, 2.564])),
        (np.array([-165.349599, 154.555399]), np.array([8.456, 5.016])),
        # Add more points and offsets as needed
    ]
    
    # Extract coordinates and offsets from the predefined list
    points = np.array([point for point, _ in points_offsets])
    offsets = np.array([offset for _, offset in points_offsets])

    # Calculate distances between the new point and the predefined points
    dists = np.sqrt(np.sum((new_point - points) ** 2, axis=1))

    # Calculate weights based on inverse distances
    weights = 1 / dists
    total_weight = np.sum(weights)

    # Calculate weighted offsets
    weighted_offset = np.sum(weights[:, np.newaxis] * offsets, axis=0) / total_weight

    return weighted_offset

# Example usage:

# New point for which we want to calculate the offsets
new_point = np.array([-93, 97])

# Calculate the offsets for the new point
offset = calculate_offsets(new_point)
print("Offset for the new point:", offset)
