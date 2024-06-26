from robodk.robolink import *      # RoboDK's API
from robodk.robomath import *      # Math toolbox for robots
from thread_MV import *

import numpy as np
import cv2
import os
import csv  
import serial
import time
import threading
import NetFT
import numpy as np
import os
from datetime import datetime


########## UPDATE MAG COORDINATES BEFORE REAL RUN

#sorting mode, determines sorting criteria. 0 = color and type, 1 = color, 2 = type
sort_mode = 1

# Define the brick search criteria. 0 = none, 1 = color, 2 = type, 3 = color and type
brick_search = 1 

# Define the target colors, types, and combinations
target_colors = ["green", "blue", "red"]  # Example colors to search for
target_types = ["2x2", "2x4"]  # Example types to search for
target_combinations = [("red", "2x2"), ("blue", "2x4")]  # Example combinations to search for


# Start the RoboDK API:
RDK = Robolink()

# Get the robot items by name:
robot = RDK.Item('UR10', ITEM_TYPE_ROBOT)

home= RDK.Item('Home')

Pick_base= RDK.Item('Pick_Base')
Place_base= RDK.Item('Place_Base')
Mag_base=RDK.Item('Mag_base')
Pick_Intermediate=RDK.Item('pick_intermediate')
Test1 = RDK.Item('test1')
Test2 = RDK.Item('test2')

World=RDK.Item('World',ITEM_TYPE_FRAME)
Ref_Pick=RDK.Item('Pick_Mat',ITEM_TYPE_FRAME)
Ref_Place=RDK.Item('Build_plate',ITEM_TYPE_FRAME)
Ref_Mags=RDK.Item('Mags',ITEM_TYPE_FRAME)
Placed_bricks = RDK.Item('Placed_bricks', ITEM_TYPE_FOLDER)

#magasin positions
mags_assigned=[[95.931,177.953],[91.841,31.130],[160,100]]

# Convert each position pair in mags_assigned to a tuple
mags_assigned = [(float(pos[0]), float(pos[1])) for pos in mags_assigned]

# Initialize brick_to_magasin as an empty dictionary
brick_to_magasin = {}

# Initialize counters for each magasin
magasin_counters = {tuple(mag): 0 for mag in mags_assigned}


# Dictionary mapping brick types to their corresponding items in RoboDK
brick_type_mapping = {
    3024: RDK.Item('LEGO1x1p', ITEM_TYPE_OBJECT),
    3023: RDK.Item('LEGO1x2p', ITEM_TYPE_OBJECT),
    3710: RDK.Item('LEGO1x4p', ITEM_TYPE_OBJECT),
    3022: RDK.Item('LEGO2x2p', ITEM_TYPE_OBJECT),
    3020: RDK.Item('LEGO2x4p', ITEM_TYPE_OBJECT),
    3005: RDK.Item('LEGO1x1b', ITEM_TYPE_OBJECT),
    3004: RDK.Item('LEGO1x2b', ITEM_TYPE_OBJECT),
    3065: RDK.Item('LEGO1x2b', ITEM_TYPE_OBJECT),
    3010: RDK.Item('LEGO1x4b', ITEM_TYPE_OBJECT),
    3066: RDK.Item('LEGO1x4b', ITEM_TYPE_OBJECT),
    3003: RDK.Item('LEGO2x2b', ITEM_TYPE_OBJECT),
    3001: RDK.Item('LEGO2x4b', ITEM_TYPE_OBJECT)
    # Add more mappings as needed
}

# Dictionary mapping brick types to their corresponding items in RoboDK
brick_offset_mapping = {
    3024: [0,0],
    3023: [0,4],
    3710: [0,12],
    3022: [4,4],
    3020: [4,12],
    3005: [0,0],
    3004: [0,4],
    3065: [0,4],
    3010: [0,12],
    3066: [0,12],
    3003: [4,4],
    3001: [4,12],
    # Add more mappings as needed
}

# Dictionary mapping instruction set color values to RGB values
color_mapping = {
    26:  ["#05131D"],        # Black
    23:  ["#0055BF"],       # Blue
    28:  ["#237841"],       # Green
    21:  ["#C91A09"],       # Red
    24:  ["#F2CD37"],       #Yellow
    1:   ["#FFFFFF"],       #White
    2:   ["#9BA19D"],       #Light Grey 
    222: ["#E4ADC8"],       #Bright Pink
    9:   ["#FC97AC"],       #Pink
    192: ["#582A12"],       #Reddish Brown
    154: ["#720E0F"],       #Dark Red
    194: ["#A0A5A9"],       #Light Bluish Grey
    308: ["#352100"],       #Dark Brown
    25:  ["#583927"],       #Brown
    29:  ["#73DCA1"],       #Meduim Green 
    151: ["#A0BCAC"],       #Sand Green 
    37:  ["#4B9F4A"],       #Bright Green
    141: ["#184632"],       #Dark Green       
    283: ["#F6D7B3"]        #Light Nougat
    # Add more mappings as needed
}

def hex_to_rgb(hex_code):
    # Remove '#' if present
    hex_code = hex_code.lstrip('#')

    # Convert hex to RGB
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def assign_magasin(brick_type, brick_color, mags_assigned, brick_to_magasin, sort_mode):
    if sort_mode == 0:
        key = f"{brick_type} {brick_color}"
    elif sort_mode == 1:
        key = brick_color
    elif sort_mode == 2:
        key = brick_type
    else:
        print("Invalid sorting mode.")
        return None
    
    print('key : '+ str(key))
    print('key type : '+str(type(key)))

    if key not in brick_to_magasin:
        if len(mags_assigned) > 0:
            magasin_position = mags_assigned.pop(0)
            brick_to_magasin[key] = magasin_position
            return magasin_position
        else:
            print("Error: No available magasins to assign the brick.")
            return None
    else:
        return brick_to_magasin[key]
    
def get_timestamped_filename(base_name, extension="csv"):
    """
    Generate a filename with the current date and time.

    :param base_name: Base name of the file.
    :param extension: File extension.
    :return: Timestamped filename.
    """
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{current_time}.{extension}"

def initialize_csv(csv_filename, header):
    """
    Initialize the CSV file with the header.

    :param csv_filename: Name of the CSV file.
    :param header: List of header names.
    """
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

def save_kpi_to_csv(kpi_array, csv_filename):
    """
    Append a KPI array as a new row in a CSV file.

    :param kpi_array: List of KPI values (1D array).
    :param csv_filename: Name of the CSV file to save the KPI data.
    """
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(kpi_array)

def ensure_directory_exists(directory):
    """
    Ensure that a directory exists, and create it if it does not.

    :param directory: Path of the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)



# Global variable to indicate if the threshold is breached
threshold_breached = False
# Function to monitor threshold

def threshold_monitor(threshold,sensor):
    global threshold_breached
    while not threshold_breached:
        # Replace this with your logic to check if the threshold is breached
        # For demonstration, we'll just use a simple counter
        #time.sleep(1)  # Simulate some processing time
        force_measurement = sensor.getForce()
        
        measure_list=[]
        for i in range(0,70):
            force_measurement = sensor.getForce()
            measure_list.append(force_measurement)
        measure_avg=[np.mean(measure_list[0]),np.mean(measure_list[1]),np.mean(measure_list[2])]
        diff_measure=calibrated_sensor_values[2]-measure_avg[2]
        if threshold[2] <= diff_measure:
            threshold_breached = True
            print("Threshold breached! Shutting down main program...")
            print("Force value difference that exceeded:",diff_measure/1000000)
            # You can perform any cleanup or shutdown procedures here
            # For demonstration, we'll just exit the program
            robot.Stop()
            os._exit(0)

def calibrate_sensor(sensor,measures=7000):
    calibrate_list=[]
    for i in range(0,measures):
        force_measurement = sensor.getForce()
        calibrate_list.append(force_measurement)
    
    return [np.mean(calibrate_list[0]),np.mean(calibrate_list[1]),np.mean(calibrate_list[2])]

def Check_for_bricks(sensor,calibrated_sensor_values,measures=700):
    measure_list=[]
    for i in range(0,measures):
        force_measurement = sensor.getForce()
        measure_list.append(force_measurement)
    measure_avg=[np.mean(measure_list[0]),np.mean(measure_list[1]),np.mean(measure_list[2])]
    diff_measure=calibrated_sensor_values[2]-measure_avg[2]
    print('Calibrated value: '+str(calibrated_sensor_values[2]/1000000))
    print('Measured value: '+ str(measure_avg[2]/1000000))
    print('This is the difference:'+str(diff_measure/1000000))

    if diff_measure/1000000 <= -0.8:
        print(" Gripper found no brick")
    elif diff_measure/1000000 >= 1.5:
        print('gripper caught on studs')
    else:
        print('gripper picked up brick')
    print("\n")
    return diff_measure


def get_single_force_measurement(sensor,measures=700):
    calibrate_list=[]
    for i in range(0,measures):
        force_measurement = sensor.getForce()
        calibrate_list.append(force_measurement)
    
    return [np.mean(calibrate_list[0]),np.mean(calibrate_list[1]),np.mean(calibrate_list[2])]


def calculate_offsets(new_x,new_y):
    new_point = [new_x,new_y]
    # Define points and their offsets
    points_offsets = [
        (np.array([-191.93,43.632]), np.array([6.056,8.349])),
        (np.array([-193.087,93.662]), np.array([8.323,8.958])),
        (np.array([-193.232,143.764]), np.array([10.233,8.167])),
        (np.array([-112.299,46.729]), np.array([6.303,5.246])),
        (np.array([-112.625,94.4]), np.array([8.885,5.595])),
        (np.array([-113.485,142.911]), np.array([11.972,7.001])),
        (np.array([-32.704,45.803]), np.array([7.763,1.838])),
        (np.array([-32.263,94.755]), np.array([10.112,2.864])),
        (np.array([-32.285,145.233]), np.array([11.897,2.56])),
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
    weighted_offset_x = weighted_offset[0]
    weighted_offset_y = weighted_offset[1]

    return weighted_offset_x, weighted_offset_y




# Get the directory of the currently executing script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path relative to the script's directory
folder_path = os.path.join(script_dir, 'Memory')
csv_file_path = os.path.join(folder_path, 'Memory data')

if os.path.exists(folder_path) and os.path.isfile(csv_file_path):
    print("Folder and CSV file exist.")
else:
    print("Folder and/or CSV file do not exist.")

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print('Folder has been created')

if not os.path.exists(csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Counter', 'Brick type', 'color'])  # Write header row
        csvwriter.writerow([0, 0, 0])  # Write initial data
    print('CSV file has been created')

with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    headers = next(csvreader)  # Read the header row
    data = next(csvreader)  # Read the data row

    # Access specific variables based on their column index
    starting_point = int(data[0])  

print(str(starting_point))
print('DATA = '+ str(data))

def generate_initial_data():
    # Extract magasins, brick_to_magasin, and mags_assigned from the sorting code context
    magasins = []  # List to store magasins data
    for magasin_id, position in enumerate(mags_assigned, start=1):
        color, brick_type = None, None
        if (brick_type, color) in brick_to_magasin:
            magasin_counter = brick_to_magasin[(brick_type, color)]
        else:
            magasin_counter = 0
        magasins.append((magasin_id, color, brick_type, magasin_counter, position))
    return magasins

def write_initial_data(csv_file_path, initial_data):
    # Check if the file exists, otherwise create it
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['MagasinID', 'Color', 'BrickType', 'Counter', 'PositionX', 'PositionY'])  # Write header row
            for magasin_id, color, brick_type, counter, position in initial_data:
                csvwriter.writerow([magasin_id, color, brick_type, counter, position[0], position[1]])  # Write initial data

#Functions:
def continue_build():
    while True:
        print('Unfinished build detetcted. do you want to continue?')
        input_var=input(" To continue build press \"y\" or to restart build press \"n\": ")
        if input_var =="y":
            print("\n")
            print("Build continuing")
            return 'resume'
        elif input_var=="n":
            print("\n")
            print("Build restarting")
            return 'restart'

        else:
            print("You have to choose yes or no by typing \"y\" or \"n\" \n")
        
        break
def tocontinue(safety):
    if safety == True:
        while True:
            input_var=input("If pickup/place is correct press \"y\": ")
            if input_var =="y":
                print("\n")
            elif input_var == 'SO':
                print('safety off')
                safety = False    
            else:
                print("You have to choose yes or no by typing \"y\" or \"n\" \n")
            
            break
    return safety 



def picked_up_brick():
    while True:
        input_var=input("Has a brick been picked up? \"y\" or \"n\": ")
        if input_var =="y":
            print("\n")
            succesful_pickup = True
        elif input_var == "n":
            print("\n")
            succesful_pickup = False
        else:
            print("You have to choose yes or no by typing \"y\" or \"n\" \n")
        
        break
    return succesful_pickup
def runmode():
    while True:
        input_var=input("To run LIVE press \"y\" or to run OFFLINE press \"n\": ")
        if input_var =="y":
            print("\n")
            print("ONLINE")
            return 1
        elif input_var=="n":
            print("\n")
            print("OFFLINE")
            return 0

        else:
            print("You have to choose yes or no by typing \"y\" or \"n\" \n")
        
        break

def pick_place(frame,x,y,z,a,b,c,speed):
    robot.setFrame(frame)
    robot.setSpeed(speed)
    pose_ref=robot.Pose()
    Pick=Mat(pose_ref)

    pos_pick=Pick.Pos()
    pos_pick=[x,y,z]
    Pick.setPos(pos_pick)
    
    #Pick=Pick*rotx(a*pi/180)*roty(b*pi/180)*rotz(c*pi/180)
    Pick=Pick*rotx(a)*roty(b)*rotz(c+(2*np.pi/180))

    robot.MoveL(Pick)

    robot.setFrame(RDK.Item('Wordl'))
    time.sleep(0.5)

#Double check I/O's
def openIO(): 
    #open
    robot.setDO(1,1) 
    robot.setDO(2,0) 
    time.sleep(0.5)

def closeIO():
    #Close
    robot.setDO(1,0) 
    robot.setDO(2,1) 
    time.sleep(0.5)

def belt_activation(message="1",COM="COM7"):
    # Replace 'COMx' with the actual COM port assigned to your USB-to-Serial converter
    serial_port = serial.Serial(COM, 115200, timeout=1)

    # Wait for the serial connection to establish
    time.sleep(2)

    # Encode the string as bytes before sending
    encoded_message = message.encode('utf-8')

    # Send the message
    serial_port.write(encoded_message)

    # Close the serial connection
    serial_port.close()
    print("closed")

def main_robot(runmode):

   
    safety = True
    pipeline,profile=initialize()
    calib_depth = cv2.imread('Robot/Calibration/average_depth_gray.png') 

    

    if runmode==1:
        # Update connection parameters if required:
        # robot.setConnectionParams('192.168.2.35',30000,'/', 'anonymous','')

        # Connect to the robot using default IP
        success = robot.Connect()  # Try to connect once
        #success robot.ConnectSafe() # Try to connect multiple times
        status, status_msg = robot.ConnectedState()
        if status != ROBOTCOM_READY:
            # Stop if the connection did not succeed
            print(status_msg)
            raise Exception("Failed to connect: " + status_msg)

        # This will set to run the API programs on the robot and the simulator (online programming)
        RDK.setRunMode(RUNMODE_RUN_ROBOT)
        # Note: This is set automatically when we Connect() to the robot through the API


        joints_ref = robot.Joints()
        print("Ready to start")
        safety = tocontinue(safety)    
    else:
        RDK.setRunMode(RUNMODE_SIMULATE)
    #TEMP
    array_ins=np.genfromtxt('Robodk_test.csv', delimiter=',')
    t=0
    global data
    global headers
    #asking if build should restart if the read counter is not 0
    if int(data[0]) != 0:
    
        if continue_build() =='restart':
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Counter', 'Brick type', 'color'])  # Write header row
                csvwriter.writerow([0, 0, 0])  # Write initial data
            with open(csv_file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader)  # Read the header row
                data = next(csvreader)  # Read the data row
            print('Build has been restarted')
            starting_point = int(data[0])
        else:
            starting_point = int(data[0])
    else:
        starting_point = 0
    print(str(starting_point))

    if int(data[0]) != 0:
    # Check if you should continue or restart the build
        if continue_build() == 'restart':
            # If you choose to restart, update the CSV file and starting_point
            with open(csv_file_path, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Counter', 'Brick type', 'color'])  # Write header row
                csvwriter.writerow([0, 0, 0])  # Write initial data

            # Read the updated data from the CSV file
            with open(csv_file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                headers = next(csvreader)  # Read the header row
                data = next(csvreader)  # Read the data row

            # Update the starting_point to 0
            starting_point = int(data[0])

            # Print a message
            print('Build has been restarted')
        else:
            # If you choose to continue, update the starting_point to the current value in data[0]
            starting_point = int(data[0])
    else:
        # If data[0] is already 0, keep starting_point as 0
        starting_point = 0

    # Print the value of starting_point
    print(str(starting_point))


    directory = 'KPI_data'

    # Ensure the directory exists
    ensure_directory_exists(directory)

    # Define the base name for the CSV file
    base_csv_name = 'KPI_DATA_sort'

    # Generate a timestamped filename and ensure it's in the desired directory
    csv_file = os.path.join(directory, get_timestamped_filename(base_csv_name))

    # Define the header for the CSV file
    kpi_header = ['brick_no','brick_color','brick_type','pickup_start','pickup_end','pickup_attempt','belt_activation','sorting_end']

    # Initialize the CSV file with the header
    initialize_csv(csv_file, kpi_header)

    KPI_Data = [None] * len(kpi_header)  # Initialize with None or default values
    # Function to update a KPI value
    def update_kpi(kpi_name, value):
        if kpi_name in kpi_header:
            index = kpi_header.index(kpi_name)
            KPI_Data[index] = value
        else:
            print(f"KPI '{kpi_name}' not found in header.")


    
    pickup_force = []

    speed_normal=150
    speed_place=10
    tool_safety=0
    safety_clearence=[0,0]
    results = []
    Calibrating=True
    roi=[0, 100, 0, 100]
    T_inpx=[0,0,0]
    px_pr_mm=0
    brick_no = 0
    with open("Robot/Calibration/Calibration_Data.json", 'r') as f:
        Calibration = json.load(f)


    
    print('starting')
    robot.setFrame(RDK.Item('Wordl'))
    robot.setSpeed(speed_normal)
    robot.setDO(0,1)
    openIO()
    robot.MoveL(home)
    first_cam = True
    safety = tocontinue(safety)

    

    while True: ########CHECK
       
      
        
        #temp comment for testing
        #xcam,ycam,Ccam=MV.get_xyA(array_ins[x,5],array_ins[x,6])
        pickup_attempt = 1
        update_kpi('pickup_attempt',pickup_attempt)
        belt_acti_index = 0
        update_kpi('belt_activation',belt_acti_index)
        brick_no += 1
        update_kpi('brick_no',brick_no)
        pickup_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_kpi('pickup_start',pickup_start)
        if  len(results) == 0:
            if not first_cam:
                belt_activation()
                belt_acti_index += 1
                update_kpi('belt_activation',belt_acti_index)
            results,roi,T_inpx,px_pr_mm=main_mv_belt(20,pipeline,profile,calib_depth,roi=roi,Calibrating=Calibrating,T_inpx=T_inpx,px_pr_mm=px_pr_mm)
            first_cam = False
        print(results)
        Calibrating=False

       
        height_target1 = (-167.0, -162)
       
        height_target2 = (-160.5, -155)
        #Initialize variables to store the position
        xcam = None
        ycam = None
        Ccam = None
        
        succesful_pickup = False
        print('succesful pickup is: ' + str(succesful_pickup))

        while not succesful_pickup:
            openIO()
            #Initialize variables to store the position
            xcam = None
            ycam = None
            Ccam = None
            Hcam = None
            # Determine the search criteria based on the value of brick_search
            if brick_search == 0:
                # Use the position of the first row
                xcam = results[0, 0]
                ycam = results[0, 1]
                Ccam = results[0, 2]
                Hcam = results[0, 8]
                color = results[0, 3]
                brick_type = results[0, 4]
                update_kpi('brick_color',color)
                update_kpi('brick_type',brick_type)
                results = np.delete(results, 0, axis=0)
            elif brick_search == 1:
                # Iterate over the results array
                for i in range(len(results)):
                    color = results[i, 3]
                    brick_type = results[i, 4]
                    depth = results[i,8]
                    if color in target_colors and (height_target1[0] <= depth <= height_target1[1] or height_target2[0] <= depth <= height_target2[1]):
                        # Store the position and exit the loop
                        xcam = results[i, 0]
                        ycam = results[i, 1]
                        Ccam = results[i, 2]
                        Hcam = results[i, 8]
                        update_kpi('brick_color',color)
                        update_kpi('brick_type',brick_type)
                        # Remove the found row from the results array
                        results = np.delete(results, i, axis=0)
                        break
            elif brick_search == 2:
                # Iterate over the results array
                for i in range(len(results)):
                    color = results[i, 3]
                    brick_type = results[i, 4]
                    depth = results[i,8]
                    if brick_type in target_types and (height_target1[0] <= depth <= height_target1[1] or height_target2[0] <= depth <= height_target2[1]):
                        # Store the position and exit the loop
                        xcam = results[i, 0]
                        ycam = results[i, 1]
                        Ccam = results[i, 2]
                        Hcam = results[i, 8]
                        update_kpi('brick_color',color)
                        update_kpi('brick_type',brick_type)
                        # Remove the found row from the results array
                        results = np.delete(results, i, axis=0)
                        break
            elif brick_search == 3:
                # Iterate over the results array
                for i in range(len(results)):
                    color = results[i, 3]
                    brick_type = results[i, 4]
                    depth = results[i,8]
                    if (color, brick_type) in target_combinations and (height_target1[0] <= depth <= height_target1[1] or height_target2[0] <= depth <= height_target2[1]):
                        # Store the position and exit the loop
                        xcam = results[i, 0]
                        ycam = results[i, 1]
                        Ccam = results[i, 2]
                        Hcam = results[i, 8]
                        update_kpi('brick_color',color)
                        update_kpi('brick_type',brick_type)

                        # Remove the found row from the results array
                        results = np.delete(results, i, axis=0)
                        break
            else:
                print("Invalid brick search criteria")

            if xcam is not None and ycam is not None and Ccam is not None and Hcam is not None:
                # Position found, use it for pickup
                print("Position found:")
                print("X:", xcam)
                print("Y:", ycam)
                print("C:", Ccam)
                print("H:", Hcam)

                offset_x,offset_y = calculate_offsets(xcam,ycam)
                #offset_x, offset_y = linear_interpolation_offsets( xcam, ycam)
                print("Offset for x:", offset_x)
                print("Offset for y:", offset_y)
                #offset_x = 0
                #offset_y = 0
                
                    #implement depth view here

                if Hcam >-161:
                    pickup_height = 9.6
                elif Hcam < -161:
                    pickup_height = 3.2
                print('pickup height is : ' + str(pickup_height))

                safety = tocontinue(safety)
                robot.MoveL(Pick_base)
                #robot.MoveL(Test2)
                
                safety = tocontinue(safety)
                pick_place(Ref_Pick,xcam+offset_x,ycam+offset_y+safety_clearence[1],pickup_height+10+tool_safety,0,0,Ccam,speed_normal)
                safety = tocontinue(safety)
                pick_place(Ref_Pick,xcam+offset_x,ycam+offset_y+safety_clearence[1],pickup_height+tool_safety-0.5,0,0,0,speed_place)
                closeIO()
                pickup_force = Check_for_bricks(sensor,calibrated_sensor_values)
                safety = tocontinue(safety)
                pick_place(Ref_Pick,xcam+offset_x,ycam+offset_y+safety_clearence[1],pickup_height+20+tool_safety,0,0,0,speed_place)

                robot.setSpeed(speed_normal)
                safety = tocontinue(safety)
                robot.MoveL(Pick_base)
                #if -0.8 <=  pickup_force <= 1.5:
                #    succesful_pickup = True
                #else:
                #    succesful_pickup = False   

                succesful_pickup = picked_up_brick()
                if not succesful_pickup:
                        pickup_attempt +=1
                        update_kpi('pickup_attempt',pickup_attempt)
                


            else:
                print("No bricks found for the specified criteria")
                safety = tocontinue(safety)
                belt_activation()
                belt_acti_index += 1
                update_kpi('belt_activation',belt_acti_index)
                robot.MoveL(home)
                safety = tocontinue(safety)
                time.sleep(8)
                results,roi,T_inpx,px_pr_mm=main_mv_belt(20,pipeline,profile,calib_depth,roi=roi,Calibrating=Calibrating,T_inpx=T_inpx,px_pr_mm=px_pr_mm)
                succesful_pickup = False
        
        #robot.MoveL(Test2)
        safety = tocontinue(safety)
        robot.MoveL(Pick_Intermediate)
        #robot.MoveL(Test1)
        safety = tocontinue(safety)
        robot.MoveL(home)

        pickup_end = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_kpi('pickup_end',pickup_end)


        #Sorting
        robot.MoveL(Mag_base)
        
        # Assign magasin and increment counter
        magasin_position = assign_magasin(brick_type, color, mags_assigned, brick_to_magasin, sort_mode)
        if magasin_position:
            # Increment counter
            magasin_counters[tuple(magasin_position)] += 1

            # Move to the assigned magasin position
            pick_place(Ref_Mags, magasin_position[0], magasin_position[1], 0, 0, 0, 0, speed_normal)
        else:
            print(f"No magasin position found or assigned for brick type {brick_type} and/or color {color}.")

        
        #Activate IO
        openIO()

        #Slow lift from place
        robot.MoveL(Mag_base)
        robot.setSpeed(speed_normal)

        initial_data = generate_initial_data()

        # Path to the CSV file
        csv_file_path = 'magasins.csv'

        # Write initial data to the CSV file
        write_initial_data(csv_file_path, initial_data)

        sort_end = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_kpi('sorting_end',sort_end)

        #save KPI to CSV
        save_kpi_to_csv(KPI_Data, csv_file)
        print(f"KPI data saved to {csv_file}")

        # Update the variable with the new value
        data[0] = str(int(data[0])+1)  # Assuming the variable is in the first column

        # Write the updated data back to the CSV file
        with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)
            csvwriter.writerow(data)

       

  


if __name__ == '__main__':
    
    ip_address = "192.168.1.1"  # Replace with the actual IP address of your sensor
    sensor = NetFT.Sensor(ip_address)
    force_values = get_single_force_measurement(sensor)

    calibrated_sensor_values=calibrate_sensor(sensor)
    threshold = [30000000, -30000000, 5*1000000]

    # Start the threshold monitor thread
    monitor_thread = threading.Thread(target=threshold_monitor, args=(threshold,sensor))
    monitor_thread.start()
    #main_function(sensor,calibrated_sensor_values)
    
    mode=runmode()
    main_robot(mode)#THIS CAN RUN LIVE
    