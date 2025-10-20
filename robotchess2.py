from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
# from smc.control.cartesian_space.cartesian_space_compliant_control import compliantMoveL
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
import pinocchio as pin

import argparse
import numpy as np
import time
import cv2


def dummy_streamer():
    """Emulate a computer vision stream that yields detection-like dicts."""
    while True:
        time.sleep(0.5)
        # Use list-of-lists for multi-detection consistency
        yield {
            "class": [1],
            "confidence": [0.9],
            "center_point": [[0.1, 0.1, 0.1]],
            "BBox": [[0.1, 0.1, 0.2, 0.2]],  # x, y, w, h in normalized units
            "data": True,
            "color_frame": np.zeros((360, 640, 3), dtype=np.uint8),
        }


def convert_coords(coords, from_file=False):
    """
    Convert camera frame coordinates to world coordinates.
    If from_file is a string path, load a 4x4 homogeneous transform from that text file.
    Otherwise use identity rotation and zero translation.
    """
    x, y, z = np.asarray(coords, dtype=float)

    if isinstance(from_file, str) and from_file:
        H = np.loadtxt(from_file, dtype=float).reshape(4, 4)
        return (H @ np.array([x, y, z, 1.0]))[:3]

    R = np.eye(3, dtype=float)
    t = np.zeros(3, dtype=float)

    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    H[:3, 3] = t
    return (H @ np.array([x, y, z, 1.0]))[:3]


def get_grip_positions(coords):
    """Return above, on, and place positions relative to the given point."""
    c = np.asarray(coords, dtype=float).copy()
    above = c + np.array([0.01, 0.0, 0.25])
    on = c + np.array([0.01, 0.0, 0.15])
    place = c + np.array([0.01, 0.0, 0.155])
    return above, on, place


def zero_robot_vel(robot, args):
    """Zero end effector spatial velocity if running on real hardware."""
    if args.real:
        robot.sendVelocityCommandToReal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Chess playing robot madness."
    parser = getClikArgs(parser)
    parser.add_argument(
        "--realsense",
        action=argparse.BooleanOptionalAction,
        help="Flag if running with realsense or not",
        default=False,
    )
    return parser.parse_args()


def show_pieces_gui(pieces):
    """
    Draw bounding boxes and labels on pieces['color_frame'] and show with OpenCV.
    Supports BBox as either normalized [x, y, w, h] or absolute [x1, y1, x2, y2].
    """
    img = pieces.get("color_frame")
    if img is None:
        return

    img = np.asarray(img)
    if img.ndim == 2:
        img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = img.copy()

    h, w = img_vis.shape[:2]
    boxes = pieces.get("BBox", [])
    classes = pieces.get("class", [])
    confs = pieces.get("confidence", [])

    for i, box in enumerate(boxes):
        x, y, a, b = box
        # Assume normalized [x, y, w, h] if a and b are in [0, 1], else treat as [x1, y1, x2, y2]
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + a) * w)
            y2 = int((y + b) * h)
        else:
            x1 = int(x)
            y1 = int(y)
            x2 = int(a)
            y2 = int(b)

        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cls_name = number2piece.get(classes[i], str(classes[i])) if i < len(classes) else "piece"
        if i < len(confs):
            label = f"{cls_name} {confs[i]:.2f}"
        else:
            label = f"{cls_name}"

        y_text = max(0, y1 - 5)
        cv2.putText(img_vis, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("Detections", img_vis)
    cv2.waitKey(1)


number2piece = {
    0: "black bishop",
    1: "black king",
    2: "black knight",
    3: "black pawn",
    4: "black queen",
    5: "black rook",
    6: "white bishop",
    7: "white king",
    8: "white knight",
    9: "white pawn",
    10: "white queen",
    11: "white rook",
}

# Fix incorrect mappings
piece2number = {
    "black bishop": 0,
    "black king": 1,
    "black knight": 2,
    "black pawn": 3,
    "black queen": 4,
    "black rook": 5,
    "white bishop": 6,
    "white king": 7,
    "white knight": 8,
    "white pawn": 9,
    "white queen": 10,
    "white rook": 11,
}


if __name__ == "__main__":
    args = get_args()
    robot = getRobotFromArgs(args)

    print("Initializing realsense stream.")
    if args.realsense:
        # Need the realsense sdk for this import.
        from computer_vision import stream_camera_frame_coords

        realsense_stream = stream_camera_frame_coords(
            multiple_pieces=True, piece_conf_thres=0.4
        )
    else:
        realsense_stream = dummy_streamer()

    robot._step()
    initial_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    if args.start_from_current_pose:
        initial_position = np.array(robot.T_w_e.translation).astype(float)
    else:
        if args.real:
            print("Use --start-from-current-pose")
            quit()
        # Do not use this one, it is inside wall.
        initial_position = np.array([0.3, 0.3, 0.5], dtype=float)

    print("Initial position of robot")
    print(initial_position)

    print("Moving to initial pose.")
    T_w_goal = pin.SE3(initial_rotation, initial_position)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)

    while True:
        try:
            pieces = next(realsense_stream)
            # Show detections
            show_pieces_gui(pieces)

            try:
                print(
                    f"Currently seeing these pieces: "
                    f"{[number2piece[p] for p in pieces['class']]}"
                )
            except Exception:
                print("Currently seeing these pieces: ", pieces)

            piece = None
            while piece is None:
                try:
                    piece = piece2number[input("Piece to move: ").lower()]
                except Exception:
                    print("Bad input.")

            command = np.array(
                [float(val) for val in input("Where to move piece: x.x,y.y: ").split(",")]
            )
            command = np.append(command, 0.0)
            print("Will move the piece this much in x and y: ", command)

            print("Looking for a chess piece using realsense camera.")
            piece_coords = None
            for _ in range(5):
                pieces = next(realsense_stream)
                show_pieces_gui(pieces)  # Update GUI as we poll
                print("Pieces", pieces)
                if piece in pieces["class"]:
                    index = pieces["class"].index(piece)
                    center = pieces["center_point"][index]
                    piece_coords = convert_coords(center, "./H.txt")
                    print("Chess piece found at: ", piece_coords)
                    break

            if piece_coords is None:
                print("Could not find your piece. Try again.")
                continue

            above, on, place = get_grip_positions(piece_coords)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            zero_robot_vel(robot, args)
            robot.openGripper()
            print("Has moved to position above the piece: ", above)

            T_w_goal = pin.SE3(initial_rotation, on)
            moveL(args, robot, T_w_goal)
            zero_robot_vel(robot, args)

            robot.closeGripper()
            time.sleep(1.0)
            print("Has moved to position on the piece and closed gripper: ", on)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            zero_robot_vel(robot, args)
            print("Has lifted the piece to", above)

            new_pos = np.asarray(piece_coords, dtype=float) + np.asarray(command, dtype=float)
            above, on, place = get_grip_positions(new_pos)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            zero_robot_vel(robot, args)
            print("Has moved the piece to above new position: ", above)

            T_w_goal = pin.SE3(initial_rotation, place)
            moveL(args, robot, T_w_goal)
            zero_robot_vel(robot, args)
            robot.openGripper()
            time.sleep(1.0)
            print("Has put down the piece at: ", place)

            T_w_goal = pin.SE3(initial_rotation, initial_position)
            moveL(args, robot, T_w_goal)
            zero_robot_vel(robot, args)
            print("Has moved back to inital pose: ", initial_position)

        except KeyboardInterrupt:
            print("Shutting down the chessbot.")
            break

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()

    # Close GUI windows on exit
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
