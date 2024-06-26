import pandas as pd

# Read the CSV file into a DataFrame
df_calibration = pd.read_csv("camera_calibration_results.csv")

# Extract the data from the DataFrame
mtx = df_calibration["Camera_Matrix"].values[0]
dist = df_calibration["Distortion_Coefficients"].values[0]
error = df_calibration["Mean_Error"].values[0]

print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist)
print("Mean error:", error)