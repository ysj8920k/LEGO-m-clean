import cv2
import os, shutil
from Bluetooth_Belt import Bluetooth_activate
import time
import serial
import pyrealsense2 as rs
import numpy as np
from datetime import datetime

arduino = serial.Serial(port='COM7', baudrate=115200, timeout=.1)
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

def post_process(frameset):
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    # Update color and depth frames:
    
    aligned_depth_frame = frameset.get_depth_frame()
    colorizer = rs.colorizer()

    #depth_frame = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    #depth_frame = frames.get_depth_frame()
    color_frame = frameset.get_color_frame()
    #if not depth_frame or not color_frame:
    #    continue
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.holes_fill, 3)
    spatial.set_option(rs.option.filter_magnitude, 5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.25)
    spatial.set_option(rs.option.filter_smooth_delta, 50)


    temporal = rs.temporal_filter()
    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)
    aligned_depth_frame = decimation.process(aligned_depth_frame)
    aligned_depth_frame = depth_to_disparity.process(aligned_depth_frame)
    aligned_depth_frame = spatial.process(aligned_depth_frame)
    aligned_depth_frame = temporal.process(aligned_depth_frame)
    aligned_depth_frame = disparity_to_depth.process(aligned_depth_frame)
    
    # Convert images to numpy arrays
    depth_image =  np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    color_image = np.asanyarray(color_frame.get_data())
    depth_image_raw = np.asanyarray(aligned_depth_frame.get_data())
    return depth_image,depth_image_raw,color_image

def capture_and_save_photo(cap,file_path='.png'):
    t=0
    while True:
        num='1'
        print('Starting Belt')
        write_read(num)

        #Bluetooth_activate("11")
        time.sleep(10)
        num='0'
        print('Stopping Belt')
        write_read(num)
        time.sleep(1)
        t+=1

        frameset =cap.wait_for_frames()

        depth_image,depth_image_raw , frame=post_process(frameset)

        
        curr_dt = datetime.now()

        # Save the captured frame as an image
        cv2.imwrite('Physical-Data-Generation/Images/'+str(int(round(curr_dt.timestamp())+t))+file_path, frame)


        print(f"Photo captured and saved as {str(t)+file_path}")

def clear_folder(folder='Physical-Data-Generation/Images/'):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def initialize():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))





    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    #config.enable_stream(rs.stream.depth)
    #config.load_settings_json
    #if device_product_line == 'L500':
    #    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    #else:
    #    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


    # Start streaming
    profile=pipeline.start(config)
    #SET EXPOSORE
    # Get the sensor once at the beginning. (Sensor index: 1)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]

    # Set the exposure anytime during the operation
    sensor.set_option(rs.option.exposure, 150.000)

    depth_sensor = profile.get_device().first_depth_sensor()
    #if depth_sensor.supports(rs.option.depth_units):
    #    depth_sensor.set_option(rs.option.depth_units, 0.0001)
    laser=False
    #if depth_sensor.supports(rs.option.emitter_enabled):
    #    depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if laser else 0.0)
    #elif laser:
    #    raise EnvironmentError('Device does not support laser')
    return pipeline,profile



if __name__ == '__main__':
    #clear_folder()
    pipeline,profile=initialize()

    #capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0 corresponds to the default camera (webcam)
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    #capture.set(cv2.CAP_PROP_EXPOSURE, -5.5 )
    # Capture and save photo with default file name 'captured_photo.jpg'


    capture_and_save_photo(pipeline)
