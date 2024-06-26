import numpy as np
import cv2 as cv
import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
profile = pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    x=900
    y=500
    depth_frame = frames.get_depth_frame()
    zDepth = depth_frame.get_distance(int(x),int(y))
    print(zDepth)