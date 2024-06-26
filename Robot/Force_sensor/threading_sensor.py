import threading
import time
import NetFT
import statistics
import numpy as np
import sys
import os

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
        if force_measurement[0]<threshold[0] < force_measurement[1] or force_measurement[2]<threshold[1] < force_measurement[3] or force_measurement[4]<threshold[2] < force_measurement[5]:
            threshold_breached = True
            print("Threshold breached! Shutting down main program...")
            print("Force values that exceeded:",force_measurement)
            # You can perform any cleanup or shutdown procedures here
            # For demonstration, we'll just exit the program
           
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

    if diff_measure>=100:
        print("I am holding a brick")
        Holding_Brick=True
    else:
        Holding_Brick=False
    print("\n")
    return Holding_Brick


def get_single_force_measurement(senor,measures=700):
    calibrate_list=[]
    for i in range(0,measures):
        force_measurement = sensor.getForce()
        calibrate_list.append(force_measurement)
    
    return [np.mean(calibrate_list[0]),np.mean(calibrate_list[1]),np.mean(calibrate_list[2])]

# Main function
def main_function(sensor,calibrated_sensor_values):
    global threshold_breached
    # Set your threshold here
    threshold = [30000000, -3000000, -300000]

    # Start the threshold monitor thread
    monitor_thread = threading.Thread(target=threshold_monitor, args=(threshold,sensor))
    monitor_thread.start()

    # Main program execution
    while not threshold_breached:
        # Replace this with your main program logic
        # For demonstration, we'll just print a message every second
        Check_for_bricks(sensor,calibrated_sensor_values)
        print("Main program running...")
        time.sleep(1)
    





# Example usage
if __name__ == "__main__":
    
    ip_address = "192.168.1.1"  # Replace with the actual IP address of your sensor
    sensor = NetFT.Sensor(ip_address)
    force_values = get_single_force_measurement(sensor)

    calibrated_sensor_values=calibrate_sensor(sensor)

    threshold=[calibrated_sensor_values[0]+30000000,calibrated_sensor_values[0]-30000000,
               calibrated_sensor_values[1]+30000000,calibrated_sensor_values[1]-30000000,
               calibrated_sensor_values[2]+30000000,calibrated_sensor_values[2]-30000000]
    #threshold = [30000000, -3000000, -300000]


    # Start the threshold monitor thread
    monitor_thread = threading.Thread(target=threshold_monitor, args=(threshold,sensor))
    monitor_thread.start()

    main_function(sensor,calibrated_sensor_values)


    
    


