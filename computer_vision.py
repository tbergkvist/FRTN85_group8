import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def detect_piece(model, img):
    results = model.predict(img, verbose=False)
    if not results:
        return None

    r0 = results[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None

    boxes = r0.boxes
    # Choose the detection with the highest confidence
    best_idx = int(boxes.conf.argmax().item())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()

    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0
    return u, v


def stream_camera_frame_coords():
    MODEL_PATH = "./chess_model.pt"

    pipeline = rs.pipeline()
    config = rs.config()
    model = YOLO(MODEL_PATH)

    # Configure streams.
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 6)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 6)

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
            uv = detect_piece(model, color)
            if uv is None:
                continue
            u, v = uv
            # Clamp to image bounds, round, and cast to int for depth lookup
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            ui = int(round(np.clip(u, 0, w - 1)))
            vi = int(round(np.clip(v, 0, h - 1)))

            # Depth at integer pixel
            Z = depth_frame.get_distance(ui, vi)
            if not Z or Z == 0.0:
                # optional: your neighborhood search here, which should also use integer indices
                continue

            # Deproject uses float pixel coordinates and metric depth
            point_3d = rs.rs2_deproject_pixel_to_point(color_intr, [float(u), float(v)], float(Z))
            X, Y, Zm = point_3d  # meters
            yield X, Y, Zm
    finally:
        pipeline.stop()

#for X, Y, Zm in stream_camera_frame_coords():
#    print(X, Y, Zm)
