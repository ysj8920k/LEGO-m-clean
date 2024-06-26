def calc_place_offset(point1, point2, expected_studs_x, expected_studs_y):
    # Define the expected distance between studs
    stud_distance = 8  # in mm

    # Calculate the expected points based on the number of studs and stud distance
    expected_studs1 = [0, 0]
    expected_studs2 = [stud_distance * expected_studs_x, stud_distance * expected_studs_y]

    # Calculate the differences between actual points and expected points
    diffx = point2[0] - expected_studs2[0]
    diffy = point2[1] - expected_studs2[1]

    # Calculate the ratio of differences for x and y
    x_ratio = diffx / expected_studs2[1]
    y_ratio = diffy / expected_studs2[0]

    return x_ratio, y_ratio

# Define the points and expected number of studs
point1 = [4, -4]
point2 = [243.509, 236.409]
expected_studs_x = 29
expected_studs_y = 31

# Calculate the offset ratios
x_ratio, y_ratio = calc_place_offset(point1, point2, expected_studs_x, expected_studs_y)
print("X Ratio:", x_ratio)
print("Y Ratio:", y_ratio)
