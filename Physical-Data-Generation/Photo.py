import cv2
import numpy as np
import pyrealsense2 as rs

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]

    # Directly pass mtx as a NumPy array
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted_img = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
    return undistorted_img

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

    depth_sensor = profile.get_device().first_depth_sensor()
    #if depth_sensor.supports(rs.option.depth_units):
    #    depth_sensor.set_option(rs.option.depth_units, 0.0001)
    laser=False
    #if depth_sensor.supports(rs.option.emitter_enabled):
    #    depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if laser else 0.0)
    #elif laser:
    #    raise EnvironmentError('Device does not support laser')
    return pipeline,profile

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

def capture_and_save_photo(file_path='captured_photo.png'):
    # Open webcam
    pipeline,profile=initialize()
    for i in range(0,10):
        frameset = pipeline.wait_for_frames()

        depth_image,depth_image_raw ,img=post_process(frameset)

    
    
    frame = img
    

    # Capture a single frame
    
     # Load mtx from CSV
    #mtx = np.genfromtxt('camera_matrix.csv', delimiter=',')

    # Load dist from CSV
    #dist = np.genfromtxt('distortion_coefficients.csv', delimiter=',')

    #frame = undistort_image(frame, mtx, dist)


    # Save the captured frame as an image
    cv2.imwrite(file_path, frame)

    

    print(f"Photo captured and saved at {file_path}")

# Capture and save photo with default file name 'captured_photo.jpg'
capture_and_save_photo()
