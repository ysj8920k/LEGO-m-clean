## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2

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



def avg_depth_colour(depth_images):


    image_data=depth_images
    # Calculate blended image
    # Calculate blended image
    dst = image_data[0]
    for i in range(len(image_data)):
        if i == 0:
            pass
        else:
            alpha = 1.0/(i + 1)
            beta = 1.0 - alpha
            dst = cv2.addWeighted(image_data[i], alpha, dst, beta,0.0)
            

            
    cv2.imwrite('average_depth_image.png',dst)
    return dst

def avg_depth_depth(depth_images):


    image_data=depth_images
        # Calculate blended image

    dst=np.mean( image_data, axis=0 )
            

    return dst
    



def depth_calibration_raw(pipeline,number_of_frames):
    depth_image_list=[]
    depth_colour_list=[]
    t=0
    while t<=number_of_frames:
        frameset = pipeline.wait_for_frames()

        depth_image,depth_image_raw ,color_image=post_process(frameset)
        depth_image_list.append(depth_image_raw)
        depth_colour_list.append(depth_image)
        t+=1

    calib_depth=avg_depth_depth(depth_image_list)
    avg_depth_colour(depth_colour_list)
    return calib_depth
"""
def compare_depth(x,y,calib_img,depth_img):
    image=depth_image
    image = cv2.circle(image, (x,y), radius=5, color=(255, 255, 255), thickness=-1)
    (b, g, r) = depth_img[x, y]
    (bc, gc, rc) = calib_img[x, y]
    print('Current')
    print(b,g,r)
    print('Calib')
    print(bc,gc,rc)
    cv2.imshow('depth', image)
    cv2.waitKey(1)
"""

def compare_depth(depth_scale,x,y,calib_raw,depth_img,depth_raw):
    image=depth_img
    image = cv2.circle(image, (x,y), radius=10, color=(255, 255, 255), thickness=-1)

    # Crop depth data:
    #depth = depth[xmin_depth:xmax_depth,ymin_depth:ymax_depth].astype(float)

    # Get data scale from the device and convert to meters
    #depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    #depth = depth * depth_scale
    #dist,_,_,_ = cv2.mean(depth)

    #img_val = depth_raw[x-10:x+10, y-10:y+10].astype(float)
    #cal_val = calib_raw[x-10:x+10, y-10:y+10].astype(float)
    img_val = depth_raw[y-10:y+10,x-10:x+10].astype(float)
    cal_val = calib_raw[y-10:y+10,x-10:x+10].astype(float)
    img_val=img_val*depth_scale
    cal_val=cal_val*depth_scale
    #dist_img,_,_,_ = cv2.mean(img_val)
    #dist_cal,_,_,_ = cv2.mean(cal_val)
    dist_img = np.mean(img_val)
    dist_cal = np.mean(cal_val)

    #print('Current')
    #print(dist_img)

    #print('Calib')
    #print(dist_cal)

    #print('diff')
    #print((dist_cal-dist_img))
    cv2.imshow('depth', image)
    cv2.waitKey(1)
    return dist_cal-dist_img

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

pipeline,profile=initialize()
try:


    calib_depth=depth_calibration_raw(pipeline,100)

    while True:

        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        dist_list=[]
        for t in range(0,10):
            # Wait for a coherent pair of frames: depth and color
            frameset = pipeline.wait_for_frames()

            depth_image,depth_image_raw, color_image=post_process(frameset)
            dist=compare_depth(depth_scale,900,500,calib_depth,depth_image,depth_image_raw)
            dist_list.append(dist)

        print('Average Depth')
        print(np.mean(dist_list)*1000)#*(1/0.00005))






        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.025), cv2.COLORMAP_JET)
        depth_colormap =  depth_image
        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        # Show images
            
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images)
        #cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()