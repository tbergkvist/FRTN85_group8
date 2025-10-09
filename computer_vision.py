import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def detect_piece(model, img):
    result = model(img)
    x1, y1, x2, y2 = result.boxes[0].xyxy[0]
    midx = (x2 - x1) / 2 + x1
    midy = (y2 - y1) / 2 + y1
    return midx, midy

def stream_camera_frame_coords():
    MODEL_PATH = "./chess_model.pt"

    pipeline = rs.pipeline()
    config = rs.config()
    model = YOLO(MODEL_PATH)

    # Configure streams.
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 10)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 10)

    # Start streaming
    profile = pipeline.start(config)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get color intrinsics for deprojection in the color frame
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()  # contains fx, fy, ppx, ppy, distortion

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            # Convert color to numpy for visualization and detection
            color = np.asanyarray(color_frame.get_data())

            # YOLO detection model to get piece pixel (u, v)
            u, v = detect_piece(model, color)

            # Get metric depth at that pixel (meters)
            Z = depth_frame.get_distance(u, v)

            # Deproject pixel to 3D using intrinsics and depth
            # Returns [X, Y, Z] in meters, camera coordinate system
            point_3d = rs.rs2_deproject_pixel_to_point(color_intr, [float(u), float(v)], float(Z))
            X, Y, Zm = point_3d  # meters
            yield X, Y, Zm
    finally:
        pipeline.stop()
