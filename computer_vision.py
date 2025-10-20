import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
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


def detect_pieces(model, img, piece_conf_thres=0.7):
    result = model.predict(img, verbose=False)[0]
    pieces = {"class": [], "confidence": [], "center_point": [], "BBox": [], "data": False}

    boxes = result.boxes  # Boxes object for bounding boxes
    if boxes is None or len(boxes) == 0:
        return pieces

    for box in boxes:
        if box.conf[0].item() < piece_conf_thres:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        midx = (x2-x1)/2 + x1
        midy = (y2-y1)/2 + y1
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())

        pieces["class"].append(class_id)
        pieces["confidence"].append(confidence)
        pieces["center_point"].append([midx, midy])
        pieces["BBox"].append([x1, y1, x2, y2])
        pieces["data"] = True
    return pieces


def pixel2coord(depth_frame, u, v, color_intr):
    # Clamp to image bounds, round, and cast to int for depth lookup
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    ui = int(round(np.clip(u, 0, w - 1)))
    vi = int(round(np.clip(v, 0, h - 1)))

    # Depth at integer pixel
    Z = depth_frame.get_distance(ui, vi)
    if not Z or Z == 0.0:
        # optional: your neighborhood search here, which should also use integer indices
        return

    # Deproject uses float pixel coordinates and metric depth
    point_3d = rs.rs2_deproject_pixel_to_point(color_intr, [float(u), float(v)], float(Z))
    return point_3d


def stream_camera_frame_coords(multiple_pieces=False, piece_conf_thres=0.7):
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

            if not multiple_pieces:
                # YOLO detection model to get piece pixel (u, v)
                uv = detect_piece(model, color)
                if uv is None:
                    continue
                u, v = uv
                X, Y, Z = pixel2coord(depth_frame, u, v, color_intr)
                yield X, Y, Z

            pieces = detect_pieces(model, color, piece_conf_thres)
            if not pieces["data"]:
                continue
            X = []
            Y = []
            Z = []
            for i in range(len(pieces["class"])):
                u, v = pieces["center_point"][i]
                x, y, z = pixel2coord(depth_frame, u, v, color_intr)
                pieces["center_point"][i] = [x, y, z]
                pieces["color_frame"] = color_frame
            yield pieces
            
    finally:
        pipeline.stop()

# for X, Y, Zm in stream_camera_frame_coords():
#     print(X, Y, Zm)

#for pieces in stream_camera_frame_coords(multiple_pieces=True):
#     print(pieces["class"])
