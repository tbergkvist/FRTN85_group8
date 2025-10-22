from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
from computer_vision import stream_camera_frame_coords
from StockFishing import best_move_local
import pinocchio as pin

import argparse
import numpy as np
import time
import cv2
import threading
import logging
import chess

""" ---------- DEFINE GLOBAL CONSTANTS ---------- """
FILES = list("ABCDEFGH")
RANKS = [str(i) for i in range(1, 9)]

""" ---------- ROBOT MOVEMENT FUNCTIONS ---------- """
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


#TODO might need to change the offset to calc from 0 instead
def get_grip_positions(coords, need_offset=False):
    """Return above, on, and place positions relative to the given point."""
    offset = 0
    if need_offset:
        offset = 0.05
    c = np.asarray(coords, dtype=float).copy()
    above = c + np.array([0.01, 0.0, 0.25+offset])
    on = c + np.array([0.01, 0.0, 0.15+offset])
    place = c + np.array([0.01, 0.0, 0.155+offset])
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
    logging.info("Has moved to position above the piece: ", above)

    T_w_goal = pin.SE3(tool_orientation, on)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.closeGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved to position on the piece and closed gripper: ", on)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has lifted the piece to", above)

    above, on, place = get_grip_positions(target_coords)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved the piece to above new position: ", above)

    T_w_goal = pin.SE3(tool_orientation, place)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    time.sleep(gripper_sleep)
    logging.info("Has put down the piece at: ", place)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved the robot back to the above position: ", above)


def move_home(tool_orientation, initial_position):
    T_w_goal = pin.SE3(tool_orientation, initial_position)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved to inital pose: ", initial_position)


def capture_piece(piece_coord, tool_orientation, is_royal=False, gripper_sleep=0.1)
    trash_coord = np.array()#TODO set value of thus    
    T_w_trash = pin.SE3(tool_orientation, trash_coord)
 
    above, on, place = get_grip_positions(piece_coords, is_royal)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    logging.info("Has moved to position above the piece: ", above)

    T_w_goal = pin.SE3(tool_orientation, on)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.closeGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved to position on the piece and closed gripper: ", on)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has lifted the piece to", above)

    moveL(args, robot, T_w_trash)
    zero_robot_vel(robot, args)
    robot.openGripper()
    time.sleep(gripper_sleep)
    robot.closeGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved the piece to the trash: ", trash_coord)


def extract_corner_coords(from_file = False):
    if from_file == False:
        return None

    corner_coords = np.fromfile(from_file)
    corner_coords.resize(4,3)
    logging.info("Corners extracted from file: ")
    for i, c in enumerate(corner_coords):
        logging.info("Corner [%.1f]: [%.3f, %.3f, %.3f]", (i+1), *c)
    return corner_coords[0], corner_coords[1], corner_coords[2], corner_coords[3]


def build_board_from_corners(corners_xy):
    """
    corners_xy: iterable of 4 (x, y) float tuples in robot base frame.
      They can be in any order.

    Returns:
      centers: dict mapping "A1".."H8" -> (x, y) center coordinates
      info:    dict with keys:
               - A1, A8, H1, H8 (corner points)
               - step_rank (vector from one rank to next)
               - step_file (vector from one file to next)
               - square_edge_mean (approx mm/cm; norm of steps)
    """
    pts = np.array(corners_xy, dtype=float)  # shape (4, 2)

    # 1) A1 = corner closest to origin (0,0)
    norms = np.linalg.norm(pts, axis=1)
    idx_A1 = int(np.argmin(norms))
    A1 = pts[idx_A1]

    # 2) Find the 2 adjacent corners and the oposite to A1
    others = np.delete(pts, idx_A1, axis=0)
    dists = np.linalg.norm(others - A1, axis=1)
    adj_idx = np.argsort(dists)
    Adj1, Adj2, Opp = others[adj_idx[0]], others[adj_idx[1]], others[adj_idx[2]]
    Adj = [Adj1, Adj2]

    # 3) Decide which adjecent is A8 vs H1:
    #    - ranks (A1->A8) align more with +X
    #    - files (A1->H1) align more with +Y
    A8 = max(Adj, key=lambda p: abs(p[0] - A1[0]))
    H1 = max(Adj, key=lambda p: abs(p[1] - A1[1]))
    H8 = Opp

    # 4) Per-square step vectors (rank/file)
    step_rank = (A8 - A1) / 8.0   # along ranks 1->8
    step_file = (H1 - A1) / 8.0   # along files A->H

    # 5) Build all 64 square centers
    centers = {}
    for r in range(8):      # rank index 0..7 (1..8) 
        for c in range(8):  # file index 0..7 (A..H) 
            center = A1 + (r + 0.5) * step_rank + (c + 0.5) * step_file
            name = f"{FILES[c]}{RANKS[r]}"
            centers[name] = (float(center[0]), float(center[1]))

    info = {
        "A1": tuple(A1), "A8": tuple(A8), "H1": tuple(H1), "H8": tuple(H8),
        "step_rank": tuple(step_rank), "step_file": tuple(step_file),
        "square_edge_mean": 0.5 * (np.linalg.norm(step_rank) + np.linalg.norm(step_file)),
    }
    return centers, info


def square_xy(name, centers):
    """Get (x,y) of a given square like 'E4' from centers dict."""
    return np.array(centers[name.upper()])


""" ----------- CHESS ENGINE FUNCTIONS ---------- """
def parse_uci_chess(uci: str):
    m = chess.Move.from_uci(uci)
    frm = chess.square_name(m.from_square)   # "e2"
    to  = chess.square_name(m.to_square)     # "e4"
    promo = m.promotion and chess.piece_symbol(m.promotion).lower()
    return frm, to, promo

def update_fen_with_uci(fen, uci, default_promo: str | None = None):
    """
    Apply a UCI move to a FEN and return (new_fen, san).
    - If the UCI lacks a promotion piece but one is required, uses default_promo (e.g., 'q').
    - Raises ValueError if the move is illegal for the given FEN.
    """

    board = chess.Board(fen)

    # If promotion is missing but needed (rare for engine moves; common for human input)
    if default_promo and len(uci) == 4:
        frm, to = uci[:2], uci[2:]
        move = chess.Move.from_uci(uci)
        # Check if this is a pawn reaching last rank → add promotion
        if board.piece_at(chess.parse_square(frm)).piece_type == chess.PAWN:
            to_sq = chess.parse_square(to)
            rank = chess.square_rank(to_sq)
            if rank in (0, 7):  # last rank
                uci = uci + default_promo.lower()  # e.g., 'q'

    move = chess.Move.from_uci(uci)

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move {uci} for FEN: {fen}")

    san = board.san(move)   # save SAN before pushing (pretty form like "exd5" or "O-O")
    board.push(move)        # updates everything: castling rights, en passant, clocks
    return board.fen(), san


def is_royal_piece(fen, first_move)
    """ Check if square is occupied på royal piece"""
    try:
        board = chess.Board(fen)
        square = chess.parse_square(first_move.lower())
        piece = board.piece_at(square)
        return piece is not None and piece.piece_type in (chess.KING, chess.QUEEN)
    except Exception: 
        return False


def is_capture_move(fen, move)
    try:
        board = chess.Board(fen)
        check_capture_move = chess.Move.from_uci(move.strip().lower())
        return board.is_capture(move)
    except Exception:
        return False


""" ---------- ARGS ---------- """
def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.set_defaults(
    robot_ip="192.168.1.150",
    plotter=False,
    gripper="onrobot",
    )
    parser.description = "Chess playing robot madness."
    parser = getClikArgs(parser)
    return parser.parse_args()


""" ----------- VISION FUNCTIONS ---------- """
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
        # Accepts a BGR frame to show
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
    
    corner_coords = extract_corner_coords("./corners.txt")
    board_coords, info = build_board_from_corners(corner_coords)
    
    logging.basicConfig(
        level=logging.INFO,                # Set the logging level
        format="%(asctime)s [%(levelname)s] %(message)s"  # Format of log messages
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
    logging.info("Initial position of robot: %s", initial_position)

    move_home(tool_orientation, initial_position)
    start_position = np.append(square_xy("D4", board_coords), 0.3)
    move_home(tool_orientation, start_position)
    try:
        while True:
            """logging.info("Looking at chess board using realsense camera.")
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
            logging.info("Will move the piece this much in x and y: %s", command)

            piece_coords = None
            if piece in pieces["class"]:
                index = pieces["class"].index(piece)
                center = pieces["center_point"][index]
                piece_coords = convert_coords(center, "./H.txt")
                logging.info("Chess piece found at: %s", piece_coords)

            if piece_coords is None:
                logging.info("Could not find your piece. Try again.")
                continue

            target_coords = np.asarray(piece_coords, dtype=float) + np.asarray(command, dtype=float)

            move_piece(piece_coords, target_coords, tool_orientation)

            move_home(tool_orientation, initial_position)"""
            
            command = input("Move piece from -> to (ex 'A1B3'): ").capitalize()
            parts = parts = [command[i:i+2] for i in range(0, len(command), 2)]            
            start_square = np.append(square_xy(parts[0], board_coords), 0.05)
            end_square = np.append(square_xy(parts[1], board_coords), 0.05)
            move_piece(start_square, end_square, tool_orientation)            
            move_home(tool_orientation, start_position)            

 
    except KeyboardInterrupt:
        logging.info("Shutting down the chessbot.")


    finally:
        # Ensure GUI thread is cleaned up
        gui.stop()

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()

