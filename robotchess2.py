from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
from computer_vision import stream_camera_frame_coords
import pinocchio as pin

import argparse
import numpy as np
import time
import cv2
import threading
import logging

COLS = list("ABCDEFGH")          # files (y direction)
ROWS = [str(i) for i in range(1, 9)]  # ranks (x direction)

def convert_coords(coords, from_file=False):
    x, y, z = coords
    H = np.fromfile(from_file)
    H.resize(4, 4)
    if from_file:
        return (H @ np.array([x, y, z, 1]))[:3]

    R = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

    t = np.array([0, 0, 0])

    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    H[:3,  3] = t
    return (H @ np.array([x, y, z, 1]))[:3]


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


def move_piece(piece_coords, target_coords, tool_orientation, gripper_sleep=1.0):
    """Picks up piece at piece_coords, moves it to target coords.
    """

    above, on, place = get_grip_positions(piece_coords)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    logging.info("Has moved to position above the piece:  [%.3f, %.3f, %.3f]", *above)

    T_w_goal = pin.SE3(tool_orientation, on)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.closeGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved to position on the piece and closed gripper:  [%.3f, %.3f, %.3f]", *on)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has lifted the piece to  [%.3f, %.3f, %.3f]", *above)

    above, on, place = get_grip_positions(target_coords)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved the piece to above new position:  [%.3f, %.3f, %.3f]", *above)

    T_w_goal = pin.SE3(tool_orientation, place)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    time.sleep(gripper_sleep)
    logging.info("Has put down the piece at:  [%.3f, %.3f, %.3f]", *place)


def move_home(tool_orientation, initial_position):
    T_w_goal = pin.SE3(tool_orientation, initial_position)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved to inital pose:  [%.3f, %.3f, %.3f]", *initial_position)

def extract_corner_coords(from_file = False):
    if from_file == False:
        return None
    
    corner_coords = np.fromfile(from_file)
    corner_coords.resize(4,3)
    logging.info("Corners extracted from file: ")
    for i, c in enumerate(corner_coords):
        logging.info("Corner [%.1f]: [%.3f, %.3f, %.3f]", (i+1), *c)
    return corner_coords[0], corner_coords[1], corner_coords[2], corner_coords[3]


def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.set_defaults(
    robot_ip="192.168.1.150",
    plotter=False,
    gripper="onrobot",
    visualizer=False,
    )
    parser.description = "Chess playing robot madness."
    parser = getClikArgs(parser)
    return parser.parse_args()


class _Gui:
    """Very small OpenCV UI that runs in a background thread."""
    def __init__(self, window_name="Detections"):
        self.window_name = window_name
        self._latest = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._th.start()

    def update(self, frame_bgr):
        with self._lock:
            self._latest = frame_bgr

    def stop(self):
        self._stop.set()
        self._th.join(timeout=1.0)
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def _loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                frame = self._latest
            if frame is not None:
                cv2.imshow(self.window_name, frame)
            # Keep UI responsive and allow closing the window or pressing q or Esc
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self._stop.set()
                break
            time.sleep(0.1)


def _render_pieces_overlay(pieces):
    """
    Draw bounding boxes and labels on pieces['color_frame'] and return BGR image.
    Supports BBox as either normalized [x, y, w, h] or absolute [x1, y1, x2, y2].
    """
    img = pieces.get("color_frame")
    if img is None:
        return None

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

    return img_vis


def show_pieces_gui(pieces, gui):
    """Render overlay and push to the GUI thread."""
    frame = _render_pieces_overlay(pieces)
    if frame is not None:
        gui.update(frame)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    args = get_args()
    robot = getRobotFromArgs(args)

    logging.info("Initializing realsense stream.")
    realsense_stream = stream_camera_frame_coords(multiple_pieces=True, piece_conf_thres=0.5)

    # Start GUI thread
    gui = _Gui("Detections")
    gui.start()

    robot._step()

    tool_orientation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    initial_position = np.array(robot.T_w_e.translation).astype(float)
    logging.info("Initial position of robot: [%.3f, %.3f, %.3f]", *initial_position)

    print("MARKUS TESTAR SAKER TA BORT, DENNA LIGGER PÅ RAD 262")
    c1,c2,c3,c4 = extract_corner_coords("./corners.txt")
    




    move_home(tool_orientation, initial_position)

    try:
        while True:
            logging.info("Looking at chess board using realsense camera.")
            pieces = next(realsense_stream)
            show_pieces_gui(pieces, gui)

            try:
                logging.info(
                    "Currently seeing these pieces: %s", [number2piece[p] for p in pieces['class']])
            except Exception:
                logging.info("Currently seeing these pieces: %s", pieces)

            piece = None
            while piece is None:
                try:
                    piece = piece2number[input("Piece to move: ").lower()]
                except Exception:
                    logging.info("Bad input.")

            command = np.array(
                [float(val) for val in input("Where to move piece: x.x,y.y: ").split(",")]
            )
            command = np.append(command, 0.0)
            logging.info("Will move the piece this much in x and y:  [%.3f, %.3f, %.3f]", *command)

            piece_coords = None
            if piece in pieces["class"]:
                index = pieces["class"].index(piece)
                center = pieces["center_point"][index]
                piece_coords = convert_coords(center, "./H.txt")
                logging.info("Chess piece found at:  [%.3f, %.3f, %.3f]", *piece_coords)

            if piece_coords is None:
                logging.info("Could not find your piece. Try again.")
                continue

            target_coords = np.asarray(piece_coords, dtype=float) + np.asarray(command, dtype=float)

            move_piece(piece_coords, target_coords, tool_orientation)

            move_home(tool_orientation, initial_position)

                
    except KeyboardInterrupt:
        logging.info("Shutting down the chessbot.")


    finally:
        gui.stop()

        if args.real:
            robot.stopRobot()

        if args.visualizer:
            robot.killManipulatorVisualizer()

        if args.save_log:
            robot._log_manager.saveLog()



#Markus is a terrorist and placing his functions down at the bottom right now 

def square_xy(square: str, A1_xy, square_size_cm=4.5):
    """
    square: like "E4" or "a1"
    A1_xy: (x0, y0) of A1 midpoint in robot base frame (same length units as square_size)
    square_size_cm: size of one square (default 4.5 cm)
    """
    square = square.strip().upper()
    if len(square) != 2 or square[0] not in "ABCDEFGH" or square[1] not in "12345678":
        raise ValueError(f"Bad square '{square}'")

    file_char = square[0]                 # A..H → y direction
    rank_char = square[1]                 # 1..8 → x direction

    file_idx = ord(file_char) - ord('A')  # 0..7
    rank_idx = int(rank_char) - 1         # 0..7

    x0, y0 = A1_xy
    s = square_size_cm

    x = x0 + rank_idx * s     # ranks increase along +x
    y = y0 + file_idx * s     # files increase along +y
    return (x, y)

def two_squares_xy(sq1, sq2, A1_xy, square_size_cm=4.5):
    p1 = square_xy(sq1, A1_xy, square_size_cm)
    p2 = square_xy(sq2, A1_xy, square_size_cm)
    return p1, p2

def nearest_square(x, y, A1_xy, square_size=4.5, clamp=True):
    """
    Return the closest square name and its indices (r,c) given a point (x,y).

    Assumptions (your mapping):
      - A1 is at A1_xy = (x0, y0)
      - Ranks 1..8 increase along +x (size = square_size)
      - Files A..H increase along +y (size = square_size)

    Params
    ------
    x, y : float
        Coordinates in same units as square_size (cm if square_size=4.5).
    A1_xy : (float, float)
        (x0, y0) of A1 midpoint in base frame.
    square_size : float
        Side length of one square (default 4.5).
    clamp : bool
        If True, clamp to board edges when (x,y) is outside 8x8.
        If False, raise ValueError when outside.

    Returns
    -------
    name : str   e.g. "E4"
    rc   : (int,int)  (rank_index, file_index) zero-based
    center_xy : (float,float) center of that square
    """
    x0, y0 = A1_xy
    # offsets in “square units”
    dx = (x - x0) / square_size   # 0 at rank 1, 7 at rank 8
    dy = (y - y0) / square_size   # 0 at file A,  7 at file H

    r = int(np.rint(dx))  # rank index 0..7
    c = int(np.rint(dy))  # file index 0..7

    if clamp:
        r = int(np.clip(r, 0, 7))
        c = int(np.clip(c, 0, 7))
    else:
        if not (0 <= r <= 7 and 0 <= c <= 7):
            raise ValueError(f"Point ({x:.2f},{y:.2f}) is off the board.")

    name = f"{COLS[c]}{ROWS[r]}"
    # exact center of that square:
    cx = x0 + (r + 0.5) * square_size
    cy = y0 + (c + 0.5) * square_size
    return name, (r, c), (cx, cy)
