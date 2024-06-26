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
from datetime import datetime

# Path to the CSV file and sensor IP address
csv_file_path_sensor = 'sensor_data.csv'
ip_address_sensor = "192.168.1.1"  # Replace with the actual IP address of your sensor
sensor = NetFT.Sensor(ip_address_sensor)

# Global flag and thread for sensor collection
sensor_collect = False
sensor_thread = None

########## UPDATE MAG COORDINATES BEFORE REAL RUN

# Define the brick search criteria. 0 = none, 1 = color, 2 = type, 3 = color and type
brick_search = 3 

# Define the target colors, types, and combinations
target_colors = ["red", "blue"]  # Example colors to search for
target_types = ["2x2", "2x4"]  # Example types to search for

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

Mag=[[0,0]]

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
    3022: [0,0], #works
    3020: [8,0],
    3005: [0,0],
    3004: [0,4],
    3065: [0,4],
    3010: [0,12],
    3066: [0,12],
    3003: [0,0], #works
    3001: [8,0],
    # Add more mappings as needed
}

brick_pickup_mapping = {
    '2x4 brick': 32,
    '2x4 plate': 32,
    '2x3 brick': 24,
    '2x3 plate': 24,
    '2x2 brick': 16,
    '2x2 plate': 16,
    '1x4 brick': 16,
    '1x4 plate': 16,
    '1x3 brick': 12,
    '1x3 plate': 12,
    '1x2 brick': 8,
    '1x2 plate': 8,
    '1x1 brick': 8,
    '1x1 plate': 8,
    # Add more brick types as needed
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
    283: ["#F6D7B3"],        #Light Nougat
    106: ["#F9BA61"],        #orange
    # Add more mappings as needed
}

def hex_to_rgb(hex_code):
    # Remove '#' if present
    hex_code = hex_code.lstrip('#')

    # Convert hex to RGB
    rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

    return rgb

# Dictionary mapping array_ins type to magasin type
type_ins_mag_mapping = {
    3001: ['2x4','brick'],
    3003: ['2x2','brick'],
    3004: ['1x2','brick'],
    3005: ['1x1','brick'],
    3010: ['1x4','brick'],
    3020: ['2x4','plate'],
    3022: ['2x2','plate'],
    3023: ['1x2','plate'],
    3024: ['1x1','plate']
}

# Dictionary mapping array_ins color to magasin color
color_ins_mag_mapping = {
    1: 'white',
    2: 'light_blue',
    9: 'pink',
    21: 'red',
    23: 'blue',
    24: 'yellow',
    25: 'brown',
    26: 'black',
    28: 'green',
    106: 'orange',

    # Add more mappings as needed
}
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

def read_magasins_csv(file_path):
    magasins = {}
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header row
        for row in csvreader:
            magasin_id = int(row[0])
            brick_type = row[1]
            color = row[2]
            counter = int(row[3])
            magasins[(brick_type, color)] = {'id': magasin_id, 'counter': counter}
    return magasins

def update_magasins_csv(file_path, magasins):
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Magasin ID', 'Brick type', 'Color', 'Counter'])  # Write header row
        for magasin_data in magasins.values():
            magasin_id = magasin_data['id']
            brick_type, color = magasin_data['brick_type'], magasin_data['color']
            counter = magasin_data['counter']
            csvwriter.writerow([magasin_id, brick_type, color, counter])

def pick_brick_from_magasin(brick_type_needed, color_needed, magasins):
    for magasin_id, magasin_data in magasins.items():
        if magasin_data['brick_type'] == brick_type_needed and magasin_data['color'] == color_needed:
            if magasin_data['counter'] > 0:
                magasin_data['counter'] -= 1
                position_coordinates = (magasin_data['position_x'], magasin_data['position_y'])  # Extract position coordinates
                return magasin_id, position_coordinates, True  # Return magasin ID, position coordinates, and True
    return None, None, False  # Return None for magasin ID and position coordinates, and False indicating failure

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

def log_sensor_data(sensor_data, brick_id):
    # Log sensor data and brick ID to CSV file
    with open(csv_file_path_sensor, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([brick_id] + sensor_data)

def collect_sensor_data(brick_id):
    global sensor_collect
    while sensor_collect:
        # Read sensor data
        sensor_data = sensor.getForce()
        
        # Log sensor data and brick ID to CSV
        log_sensor_data(sensor_data, brick_id)
        
        # Pause for 1 second before next reading
        time.sleep(0.05)

def start_sensor_collection(brick_id):
    global sensor_collect, sensor_thread
    sensor_collect = True
    sensor_thread = threading.Thread(target=collect_sensor_data, args=(brick_id,), daemon=True)
    sensor_thread.start()

def stop_sensor_collection():
    global sensor_collect
    sensor_collect = False
    if sensor_thread:
        sensor_thread.join()

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
        for i in range(0,200):
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

    if diff_measure/1000000 <= -1.6:
        print(" Gripper found no brick")
    elif diff_measure/1000000 >= 2.2:
        print('gripper caught on studs')
    else:
        print('gripper picked up brick')
    print("\n")
    return diff_measure


def get_single_force_measurement(senor,measures=700):
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

def calc_place_offset(x,y):
    
    # Define the points
    point1 = [-34.5,-30]
    point2 = [-24.75,320.5]
    point3 = [316,-40.75]

    stx = 44
    sty = 44
     
    # Define the expected distance between studs
    stud_distance = 8  # in mm

    lx = point3[0]-point1[0]
    ly = point2[1]-point1[1]
    #print('ly: ' + str(ly))
    #print('lx: ' + str(lx))


    dx = (lx-stud_distance*stx)/(stud_distance*stx)
    dy = (ly-stud_distance*sty)/(stud_distance*sty)
    dxy = (point2[0]-point1[0])/(stud_distance*sty)
    dyx = (point3[1]-point1[1])/(stud_distance*stx)
    #print('dx,dy,dxy,dyx:')
    #print(str(dx))
    #print(str(dy))
    #print(str(dxy))
    #print(str(dyx))

    x_new = x-0.8 + ((x+point1[0])*(dx+0.002)) + ((y+point1[1])*(dxy-0.0045))
    y_new = y-0.6 + ((y+point1[1])*(dy-0.005)) + ((x+point1[0])*(dyx+0.0045))

    return x_new,y_new


def pick_place(frame,x,y,z,a,b,c,speed):
    robot.setFrame(frame)
    robot.setSpeed(speed)
    pose_ref=robot.Pose()
    Pick=Mat(pose_ref)

    pos_pick=Pick.Pos()
    pos_pick=[x,y,z]
    Pick.setPos(pos_pick)
    
    Pick=Pick*rotx(a)*roty(b)*rotz(c+(2*np.pi/180))

    robot.MoveL(Pick)

    robot.setFrame(RDK.Item('Wordl'))
    time.sleep(0.2)

def movetype_place(frame, x, y, z, a, b, c, speed, mtype):
    robot.setFrame(frame)
    robot.setSpeed(speed)
    pose_ref = robot.Pose()
    Pick = Mat(pose_ref)

    # ending location
    pos_pick = Pick.Pos()
    pos_pick = [x, y, z]
    Pick.setPos(pos_pick)

    Pick = Pick * rotx(a) * roty(b) * rotz(c)

    # movement type location
    if mtype == 0:
        pose_ref = robot.Pose()
        Pick_move = Mat(pose_ref)
        offset_x = 0
        offset_y = 0
        print('movement type is : ' + str(mtype))

    elif 1 <= mtype <= 8:
        angle_rad = np.pi - ((mtype - 1) * (pi / 4))  # Angle in radians (45 degrees increments)
        offset_x = 1 * np.cos(angle_rad)
        offset_y = 1 * np.sin(angle_rad)
        print('movement type is : ' + str(mtype))
        angle_deg = math.degrees(angle_rad)
        print('movetype angle is : ' +str(angle_deg))

        pose_ref = robot.Pose()
        Pick_move = Mat(pose_ref)

        # ending location with offset
        pos_pick = Pick.Pos()
        pos_pick = [x - 3*offset_x, y - 3*offset_y, z]
        Pick_move.setPos(pos_pick)
        Pick_move = Pick_move * rotx(a) * roty(b) * rotz(c)

        robot.MoveL(Pick_move)

    elif mtype == 9:
        # Square spiral pattern
        spiral_offsets = [
            [1, 1], [1, -1], [-1, -1], [-1, 1],
            [0.5, 1], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0,0]
            # Add more offsets as needed
        ]

        # Move to each offset in the spiral pattern
        for offset_x, offset_y in spiral_offsets:
            pose_ref = robot.Pose()
            Pick_move = Mat(pose_ref)

            # ending location with offset
            pos_pick = Pick.Pos()
            pos_pick = [x + offset_x, y + offset_y, z]
            Pick_move.setPos(pos_pick)
            Pick_move = Pick_move * rotx(a) * roty(b) * rotz(c)

            robot.MoveL(Pick_move)

        # After moving to all offsets, return to the initial frame
        robot.setFrame(RDK.Item('Wordl'))
        time.sleep(0.2)
        # Exit the function since we've already moved the robot
        return

    else:
        print(f"Movement type {mtype} is not supported.")
        return  # Exit the function if the movement type is not supported

    # calculate the offset
    pose_ref = robot.Pose()
    Pick_move = Mat(pose_ref)

    # ending location with offset
    pos_pick = Pick.Pos()
    pos_pick = [x + offset_x, y + offset_y, z]
    Pick_move.setPos(pos_pick)
    Pick_move = Pick_move * rotx(a) * roty(b) * rotz(c)

    robot.MoveL(Pick_move)

    robot.setFrame(RDK.Item('Wordl'))
    time.sleep(0.2)

#Double check I/O's
def openIO(): 
    #open
    robot.setDO(1,1) 
    robot.setDO(2,0)
    time.sleep(0.2) 

def closeIO():
    #Close
    robot.setDO(1,0) 
    robot.setDO(2,1)
    time.sleep(0.2) 

def belt_activation(message="1",COM="COM4"):
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

    pipeline,profile=initialize()
    calib_depth = cv2.imread('Robot/Calibration/average_depth_gray.png') 
    safety = True

    if runmode==1:
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
    array_ins=np.genfromtxt('cat.csv', delimiter=',')
    print('array_ins:')
    print(array_ins)
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
    base_csv_name = 'KPI_DATA_build'

    # Generate a timestamped filename and ensure it's in the desired directory
    csv_file = os.path.join(directory, get_timestamped_filename(base_csv_name))

    # Define the header for the CSV file
    kpi_header = ['brick_id','pickup_start', 'pickup_end', 'pickup_attempts', 'belt_activation','place_end']

    # Initialize the CSV file with the header
    initialize_csv(csv_file, kpi_header)

    KPI_Data = [None] * len(kpi_header)  # Initialize with None or default values
    KPI_Data = [None] * len(kpi_header)  # Initialize with None or default va
    # Function to update a KPI value
    def update_kpi(kpi_name, value):
        if kpi_name in kpi_header:
            index = kpi_header.index(kpi_name)
            KPI_Data[index] = value
        else:
            print(f"KPI '{kpi_name}' not found in header.")

    speed_normal=200
    speed_place=10
    tool_safety=0
    safety_clearence=[0,0]
    results = []
    Calibrating=True
    roi=[0, 100, 0, 100]
    T_inpx=[0,0,0]
    px_pr_mm=0
    
    with open("Robot/Calibration/Calibration_Data.json", 'r') as f:
        Calibration = json.load(f)
    
    


    
    print('starting')
    robot.setFrame(RDK.Item('Wordl'))
    robot.setSpeed(speed_normal)
    robot.setDO(0,1)
    openIO()
    openIO()
    robot.MoveL(home)
    first_cam = True

    for q in range(starting_point,len(array_ins[:,0])-1): ########CHECK
        x=q+1
        global brick_id
        brick_id = int(array_ins[x,4])
        update_kpi('brick_id',brick_id)
        print('looking for brick of type ' + str(array_ins[x,5]) + ' and color '+ str(array_ins[x,6]))
        #pickup type alternates between pikcing up from conveyer (0) and picking from magasin (1)
        ################ EXPERIMENTAL!!! DO NOT RUN MAGASIN UNTIL POSITIONS HAVE BEEN DEFINED
        pickup_type=0
        pickup_attempt = 1
        update_kpi('pickup_attempts',pickup_attempt)
        belt_acti_index = 0
        update_kpi('belt_activation',belt_acti_index)
        pickup_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_kpi('pickup_start',pickup_start)

        if pickup_type == 0:
            if  len(results) == 0:
                if not first_cam:
                    belt_activation()
                    time.sleep(4)
                    belt_acti_index += 1
                    update_kpi('belt_activation',belt_acti_index)
                results,roi,T_inpx,px_pr_mm=main_mv_belt(15,pipeline,profile,calib_depth,roi=roi,Calibrating=Calibrating,T_inpx=T_inpx,px_pr_mm=px_pr_mm)
                first_cam = False
            print(results)
            Calibrating=False
           
           

            type_target = type_ins_mag_mapping[int(array_ins[x,5])][0]
            color_target = color_ins_mag_mapping[int(array_ins[x,6])]
            target_combinations = [(color_target, type_target)]
            brick_height = type_ins_mag_mapping[int(array_ins[x,5])][1]

            if brick_height == 'plate':
                height_target = (-167.0, -162)
            elif brick_height == 'brick':
                height_target = (-160.5, -155)


            succesful_pickup = False
            #print('succesful pickup is: ' + str(succesful_pickup))

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
                elif brick_search == 1:
                    # Iterate over the results array
                    for i in range(len(results)):
                        color = results[i, 3]
                        if color in target_colors:
                            # Store the position and exit the loop
                            xcam = results[i, 0]
                            ycam = results[i, 1]
                            Ccam = results[i, 2]
                            Hcam = results[i, 8]
                            # Remove the found row from the results array
                            results = np.delete(results, i, axis=0)
                            break
                elif brick_search == 2:
                    # Iterate over the results array
                    for i in range(len(results)):
                        brick_type = results[i, 4]
                        if brick_type in target_types:
                            # Store the position and exit the loop
                            xcam = results[i, 0]
                            ycam = results[i, 1]
                            Ccam = results[i, 2]
                            Hcam = results[i, 8]
                            # Remove the found row from the results array
                            results = np.delete(results, i, axis=0)
                            break
                elif brick_search == 3:
                    # Iterate over the results array
                    for i in range(len(results)):
                        color = results[i, 3]
                        brick_type = results[i, 4]
                        depth = results[i,8]
                        if (color, brick_type) in target_combinations and height_target[0] <= depth <= height_target[1]:
                            # Store the position and exit the loop
                            xcam = results[i, 0]
                            ycam = results[i, 1]
                            Ccam = results[i, 2]
                            Hcam = results[i, 8]
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
                    pick_place(Ref_Pick,xcam+offset_x,ycam+offset_y+safety_clearence[1],pickup_height+tool_safety-1,0,0,0,speed_place)
                    closeIO()
                    time.sleep(0.25)
                    pickup_force = Check_for_bricks(sensor,calibrated_sensor_values)/1000000
                    time.sleep(0.25)
                    safety = tocontinue(safety)
                    pick_place(Ref_Pick,xcam+offset_x,ycam+offset_y+safety_clearence[1],pickup_height+20+tool_safety,0,0,0,speed_place)

                    robot.setSpeed(speed_normal)
                    safety = tocontinue(safety)
                    robot.MoveL(Pick_base)

                    if -1.6 <=  pickup_force <= 2.2:
                        succesful_pickup = True
                    else:
                        succesful_pickup = False   
                    #succesful_pickup = picked_up_brick()
                    if not succesful_pickup:
                        pickup_attempt +=1
                        update_kpi('pickup_attempts',pickup_attempt)
                        


                else:
                    print("No bricks found for the specified criteria")
                    safety = tocontinue(safety)
                    belt_activation()
                    belt_acti_index += 1
                    update_kpi('belt_activation',belt_acti_index)
                    robot.MoveL(home)
                    time.sleep(4)
                    safety = tocontinue(safety)
                    results,roi,T_inpx,px_pr_mm=main_mv_belt(15,pipeline,profile,calib_depth,roi=roi,Calibrating=Calibrating,T_inpx=T_inpx,px_pr_mm=px_pr_mm)
                    succesful_pickup = False

                

                
            
            safety = tocontinue(safety)
            robot.MoveL(home)

        elif pickup_type == 1:   
            magasins_file_path = 'magasins.csv'
            magasins = read_magasins_csv(magasins_file_path)

          
            array_ins_brick_type = array_ins[x][5]  # Extracting brick type from array_ins
            array_ins_color = array_ins[x][6]  # Extracting color from array_ins

            # Translate array_ins brick type to magasin brick type
            brick_type_needed = type_ins_mag_mapping[array_ins_brick_type]

            # Translate array_ins color to magasin color
            color_needed = color_ins_mag_mapping[array_ins_color]

            # Pick a brick from the magasin if available
            magasin_id, position_coordinates, success = pick_brick_from_magasin(brick_type_needed, color_needed, magasins)
            if success:
                print(f"Picked a {color_needed} {brick_type_needed} from magasin {magasin_id}.")
                print("Position coordinates:", position_coordinates)

                # Calculate the pickup offset based on the brick type and the counter for the magasin
                pickup_offset = magasins[(brick_type_needed, color_needed)]['counter'] * brick_pickup_mapping.get(brick_type_needed, 0)
                print(pickup_offset)

                safety = tocontinue(safety)
                #UPDATE THE POSITIONS AND ADD THE PICKUP OFFSET
                pick_place(Ref_Pick,position_coordinates[0],position_coordinates[1],12.3+10+tool_safety,0,0,Ccam+3.2,speed_normal)
                safety = tocontinue(safety)
                pick_place(Ref_Pick,position_coordinates[0],position_coordinates[1],12.3+tool_safety,0,0,0,speed_place)
                closeIO()
                safety = tocontinue(safety)
                pick_place(Ref_Pick,position_coordinates[0],position_coordinates[0],12.3+10+tool_safety,0,0,0,speed_place)
                
                # Update the magasins CSV file
                update_magasins_csv(magasins_file_path, magasins)
            else:
                print(f"No available {color_needed} {brick_type_needed} in the magasins.")
        pickup_end = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_kpi('pickup_end',pickup_end)


        print('brick type is : ' + str(array_ins[x,5]))
        #if int(array_ins[x,5]) in {3023,3710,3022,3020,3024}:
        #    PlaceZ=3.2
        #else:
        #    PlaceZ=9.6
        PlaceZ = 3.2
        print('place Z is : '+str(PlaceZ))
        
        surround_variable = int(array_ins[x,8])  

        if int(array_ins[x,7]) == 9:
            surround_height = 3+((surround_variable-1)*3.2)
        else:
            surround_height=(surround_variable*3.2)/1.5
        
        movetypeheight = max(2.7,surround_height)
        print('move type height : '+ str(movetypeheight))

        global BrickTypeOffset
        
        brick_type_key = array_ins[x,5]

        if brick_type_key in brick_offset_mapping:
            BrickTypeOffset = brick_offset_mapping[brick_type_key]
            print('brick type offset is : '+str(BrickTypeOffset))
        else:
            print(f"No offset mapping found for brick type {brick_type_key}")

        # Apply rotation to the offset using 2D rotation formula
        
        #if array_ins[x,3] == 0:
        #    rotated_offset_x = -BrickTypeOffset[0] + (array_ins[x,1]*x_ratio)
        #    rotated_offset_y = -BrickTypeOffset[1] + (array_ins[x,0]*y_ratio)
        #elif array_ins[x,3] ==1:
        #    rotated_offset_x = -BrickTypeOffset[1] + (array_ins[x,1]*x_ratio)
        #    rotated_offset_y = -BrickTypeOffset[0] + (array_ins[x,0]*y_ratio)
        csv_type = 'STL'

        if csv_type == 'LXFML':
            if array_ins[x,3] == 1:
                rotation_offset= [0+BrickTypeOffset[0]-16,BrickTypeOffset[1]-8] 
            elif array_ins  [x,3] == 0:
                rotation_offset = [4-BrickTypeOffset[1],-4+BrickTypeOffset[0]] 
            else:
                print('Unsuported rotation given!')
                break
        elif csv_type == 'STL':
            if array_ins[x,3] == 1:
                rotation_offset= [0+BrickTypeOffset[0]-16,BrickTypeOffset[1]-8] 
            elif array_ins  [x,3] == 0:
                rotation_offset = [4-BrickTypeOffset[1]-16-8+2,-4+BrickTypeOffset[0]-8-1.5] 
            else:
                print('Unsuported rotation given!')
                break
        else:
            print('csv type of [' + str(csv_type) + '] is not supported')
            robot.Stop()
        

        rotated_offset_x = array_ins[x,0] + rotation_offset[0]
        rotated_offset_y = array_ins[x,1] + rotation_offset[1]

        print('rotated offset is : ' + str(rotated_offset_x) + ' and ' + str(rotated_offset_y))
        #LXFML instructions
        safety = tocontinue(safety)
        robot.MoveL(Place_base)
        
        global sensor_collect

        #Placing
        print('Placing:')
        print(array_ins[x,:])
        x_place,y_place = calc_place_offset(rotated_offset_x,rotated_offset_y)
        
        #x_place = x_place + rotation_offset[0]
        #y_place = y_place + rotation_offset[1]
        print('trying to place in position ' + str(x_place) + ' , ' + str(y_place))

        safety = tocontinue(safety)
        angle=-90 
        #osition 25 mm above place position
        pick_place(Ref_Place,x_place,y_place,array_ins[x,2]+PlaceZ+tool_safety+25,0,0,angle*(np.pi/180)+array_ins[x,3]*(np.pi/2),speed_normal)
        start_sensor_collection(brick_id)
        sensor_collect = True
        safety = tocontinue(safety)#execute the movement type from 20 mm above to movetypeheight mm
        if int(array_ins[x,7]) == 9:
            print('brick spiral')
            movetype_place(Ref_Place,x_place,y_place,array_ins[x,2]+PlaceZ+tool_safety+9.3,0,0,0,speed_place,array_ins[x,7])
            print('2 plate spiral')
            movetype_place(Ref_Place,x_place,y_place,array_ins[x,2]+PlaceZ+tool_safety+6.1,0,0,0,speed_place,array_ins[x,7])
            print('1 plate spiral')
            movetype_place(Ref_Place,x_place,y_place,array_ins[x,2]+PlaceZ+tool_safety+2.9,0,0,0,speed_place,array_ins[x,7])
        else:
            movetype_place(Ref_Place,x_place,y_place,array_ins[x,2]+PlaceZ+tool_safety+movetypeheight,0,0,0,speed_place,array_ins[x,7])
        safety = tocontinue(safety)
        #placeing straight down
        pick_place(Ref_Place,x_place,y_place,array_ins[x,2]+PlaceZ+tool_safety-3,0,0,0,speed_place)

        #Activate IO
        openIO()



        # Get the brick type from your instruction set
        brick_type = int(array_ins[x,5])
        print('Brick type is ' + str(brick_type))

        # Map the brick type to the corresponding item in RoboDK
        brick_item = brick_type_mapping.get(brick_type, None)

        # Check if the brick item exists
        if brick_item is None:
            # Default item if type is not found in the mapping
            print('brick type is not supported''/n''Please ensure that the brick type is proberly implemented and try again')

       
        #brick position
        position = [x_place,y_place,array_ins[x,2]+PlaceZ]
        print(str(position))
        rotation= angle*(np.pi/180)-array_ins[x,3] * (np.pi/2) - (2*np.pi/180)
        print(str(rotation))
        # Define a unique variable name using the "brick + x" naming convention
        brick_name = 'brick' + str(x)

        brick_item.Copy()
        
        # Duplicate the LEGO brick and assign it to the unique variable
        exec(brick_name + ' = RDK.Paste()')

        # Set the position for the duplicate using the specific position
        exec(brick_name + '.setPose(transl(position[0], position[1], position[2]) * rotz(rotation) * rotx(pi/2))')
        
        # Set the existing reference frame as the parent of the duplicated brick
        exec(brick_name + '.setParent(Ref_Place)')

        #naming the bricks in robodk
        robodk_name='placed'+str(x)
        exec(brick_name + '.setName(robodk_name)')

        # Get the color value from your instruction set
        color_value = int(array_ins[x,6]) 

        # Map the color value to the corresponding RGB value using the dictionary
        if color_value in color_mapping:
            hex = color_mapping[color_value]
            print('hexcolor is '+ str(hex))
            color_rgb=hex_to_rgb(hex[0])
        else:
            # Default color if not found in the mapping
            color_rgb = [0, 0, 0]

        # Normalize RGB values to the range 0-1
        normalized_color = [value / 255.0 for value in color_rgb]
        # Set the color of the duplicated brick
        exec(brick_name + '.Recolor(normalized_color)')

        #adding the bricks to a folder for oginazation
        exec(brick_name + '.setParent(Placed_bricks)')

        #Slow lift from place
        safety = tocontinue(safety)
        pick_place(Ref_Place,x_place,y_place,array_ins[x,2]+15+tool_safety,0,0,0,speed_place)
        stop_sensor_collection()
        safety = tocontinue(safety)
        robot.setSpeed(speed_normal)
        robot.MoveL(Place_base)

        place_end = datetime.now().strftime("%Y%m%d_%H%M%S")
        update_kpi('place_end',place_end)

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

       

    robot.setFrame(RDK.Item('Wordl'))
    robot.setSpeed(speed_normal)
    safety = tocontinue(safety)
    robot.MoveL(home)
    print('done')
    with open(csv_file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Counter', 'Brick type', 'color'])  # Write header row
            csvwriter.writerow([0, 0, 0])  # Write initial data
            print('Memory has been cleared')



if __name__ == '__main__':
    
    ip_address = "192.168.1.1"  # Replace with the actual IP address of your sensor
    sensor = NetFT.Sensor(ip_address)
    force_values = get_single_force_measurement(sensor)

    calibrated_sensor_values=calibrate_sensor(sensor)
    threshold = [30000000, -30000000, 7*1000000]

    #Start the threshold monitor thread
    monitor_thread = threading.Thread(target=threshold_monitor, args=(threshold,sensor))
    monitor_thread.start()
    #main_function(sensor,calibrated_sensor_values)

    mode=runmode()
    main_robot(mode)#THIS CAN RUN LIVE

