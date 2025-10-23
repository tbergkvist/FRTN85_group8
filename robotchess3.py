from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
from computer_vision import stream_camera_frame_coords
from StockFishing import best_move_local

import pinocchio as pin
import argparse
import numpy as np
import time
import logging
import chess


""" ---------- DEFINE GLOBAL CONSTANTS ---------- """
FILES = list("abcdefgh")
RANKS = [str(i) for i in range(1, 9)]
global start_position

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


def get_grip_positions(coords, royal_offset=False):
    """Return above, on, and place positions relative to the given point."""
    offset = 0.025 if royal_offset else 0

    c = np.asarray(coords, dtype=float).copy()
    above = c + np.array([0.01, 0.0, 0.25 + offset])
    on = c + np.array([0.01, 0.0, 0.15 + offset])
    place = c + np.array([0.01, 0.0, 0.155 + offset])
    return above, on, place


def zero_robot_vel(robot, args):
    """Zero end effector spatial velocity if running on real hardware."""
    if args.real:
        robot.sendVelocityCommandToReal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def move_piece(piece_coords, target_coords, tool_orientation, is_royal=False, gripper_sleep=1.0):
    """Picks up piece at piece_coords, moves it to target coords.
    """
    above, on, place = get_grip_positions(piece_coords, is_royal)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    logging.info("Has moved to position above the piece: [%.3f, %.3f, %.3f]", *above)

    T_w_goal = pin.SE3(tool_orientation, on)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.closeGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved to position on the piece and closed gripper: [%.3f, %.3f, %.3f]", *on)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has lifted the piece to [%.3f, %.3f, %.3f]", *above)

    above, on, place = get_grip_positions(target_coords, is_royal)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved the piece to above new position: [%.3f, %.3f, %.3f]", *above)

    T_w_goal = pin.SE3(tool_orientation, place)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    time.sleep(gripper_sleep)
    logging.info("Has put down the piece at: [%.3f, %.3f, %.3f]", *place)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved the robot back to the above position: [%.3f, %.3f, %.3f]", *above)


def move_home(initial_position, tool_orientation):
    T_w_goal = pin.SE3(tool_orientation, initial_position)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has moved to inital pose: [%.3f, %.3f, %.3f]", *initial_position)


def capture_piece(piece_coord, tool_orientation, is_royal=False, gripper_sleep=1.0, trash_coord=np.array([0.4, -0.1, 0.3])):
    """Picks up piece at piece_coords, throws it in trash.""" 
 
    above, on, place = get_grip_positions(piece_coord, is_royal)
    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.openGripper()
    logging.info("Has moved to position above the piece: [%.3f, %.3f, %.3f]", *above)

    T_w_goal = pin.SE3(tool_orientation, on)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    robot.closeGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved to position on the piece and closed gripper: [%.3f, %.3f, %.3f]", *on)

    T_w_goal = pin.SE3(tool_orientation, above)
    moveL(args, robot, T_w_goal)
    zero_robot_vel(robot, args)
    logging.info("Has lifted the piece to [%.3f, %.3f, %.3f]", *above)
    T_w_home = pin.SE3(tool_orientation, start_position)
    moveL(args, robot, T_w_home)
    zero_robot_vel(robot, args)
    T_w_trash = pin.SE3(tool_orientation, trash_coord)
    moveL(args, robot, T_w_trash)
    zero_robot_vel(robot, args)
    robot.openGripper()
    time.sleep(gripper_sleep)
    logging.info("Has moved the piece to the trash: [%.3f, %.3f, %.3f]", *trash_coord)


def extract_corner_coords(from_file=False):
    """Extract the corner coordinates in the robot frame from the file."""
    if not from_file:
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
    step_rank = (step_rank / np.linalg.norm(step_rank)) * 0.045
    step_file = (step_file / np.linalg.norm(step_file)) * 0.045
        
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
    return np.array(centers[name.lower()])


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


def is_royal_piece(fen, first_move):
    """ Check if square is occupied på royal piece."""
    try:
        board = chess.Board(fen)
        square = chess.parse_square(first_move.lower())
        piece = board.piece_at(square)
        return piece is not None and piece.piece_type in (chess.KING, chess.QUEEN)
    except Exception: 
        return False


def is_capture_move(fen, move):
    """Check if move captures other piece."""
    try:
        board = chess.Board(fen)
        check_capture_move = chess.Move.from_uci(move.strip().lower())
        return board.is_capture(check_capture_move)
    except Exception:
        return False

def is_casteling_move(fen: str, uci: str):
    """
    Heuristic: detects castling from UCI + FEN (standard chess).
    Returns:
      (is_castling: bool, king_uci: str|None, rook_uci: str|None, side: 'white'|'black'|None, kind: 'kingside'|'queenside'|None)

    Example success: (True, 'e1g1', 'h1f1', 'white', 'kingside')
    Example failure: (False, None, None, None, None)
    """
    try:
        board = chess.Board(fen)
        m = chess.Move.from_uci(uci.strip().lower())
    except Exception:
        return False, None, None, None, None

    frm, to = m.from_square, m.to_square
    piece = board.piece_at(frm)

    # Must be a king of the side to move
    if piece is None or piece.piece_type != chess.KING or piece.color != board.turn:
        return False, None, None, None, None

    # Same rank and king moved exactly two files -> castling pattern
    same_rank = chess.square_rank(frm) == chess.square_rank(to)
    two_files = abs(chess.square_file(frm) - chess.square_file(to)) == 2
    if not (same_rank and two_files):
        return False, None, None, None, None

    side = "white" if piece.color == chess.WHITE else "black"
    kingside = chess.square_file(to) > chess.square_file(frm)
    kind = "kingside" if kingside else "queenside"

    # Build king and rook UCI moves
    king_from = chess.square_name(frm)
    king_to   = chess.square_name(to)

    if piece.color == chess.WHITE:
        rook_from = "h1" if kingside else "a1"
        rook_to   = "f1" if kingside else "d1"
    else:
        rook_from = "h8" if kingside else "a8"
        rook_to   = "f8" if kingside else "d8"

    king_uci = f"{king_from}{king_to}"
    rook_uci = f"{rook_from}{rook_to}"
    return True, king_uci, rook_uci, side, kind

def chess_coord_to_robot_coord(chess_move,board_coords):
    return np.append(square_xy(chess_move,board_coords), 0.05)

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


if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

    args = get_args()
    robot = getRobotFromArgs(args)

    logging.info("Initializing realsense stream.")
    realsense_stream = stream_camera_frame_coords(multiple_pieces=True, piece_conf_thres=0.5)

    robot._step()

    tool_orientation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )

    corner_coords = extract_corner_coords("./corners.txt")
    board_coords, info = build_board_from_corners(corner_coords)
    logging.info("Board info: ")
    print(info)
    initial_position = np.array(robot.T_w_e.translation).astype(float)
    logging.info("Initial position of robot: [%.3f, %.3f, %.3f]", *initial_position)
    move_home(initial_position, tool_orientation)
    
    start_position = np.append(square_xy("D4", board_coords), 0.3)
    move_home(start_position, tool_orientation)

    #starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" #this one has casteling rights for both kings
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1" #default fen currently casteling removed. 
    current_fen = starting_fen
    

    """Is this good to have or should we refactor?"""
    """Chose mode for the code, manual or robot."""        
    robot_mode = int(input("Manual mode or Robot solo: (1 or 2) "))
    
    try:
        if robot_mode == 1:
            while True:
                command = input("Move piece from -> to (ex 'a1b3'): ").lower()
                parts = parts = [command[i:i+2] for i in range(0, len(command), 2)]            
                
                start_square = np.append(square_xy(parts[0], board_coords), 0.05)
                end_square = np.append(square_xy(parts[1], board_coords), 0.05)
                start_square = chess_coord_to_robot_coord(parts[0],board_coords) #NU KAN DU TA BORT DOM GAMLA
                end_square = chess_coord_to_robot_coord(parts[1],board_coords)
                
                move_piece(start_square, end_square, tool_orientation)            
                move_home(start_position, tool_orientation)            

        elif robot_mode == 2:
            while True:
                command = input("Move piece from -> to (ex 'a1b3'): ").lower()
                parts = [command[i:i+2] for i in range(0, len(command), 2)]            
                    

                start_square = np.append(square_xy(parts[0], board_coords), 0.05) 
                end_square = np.append(square_xy(parts[1], board_coords), 0.05)
                
                start_square = chess_coord_to_robot_coord(parts[0],board_coords) # NU KAN DU TA BORT DOM GAMLA 
                end_square = chess_coord_to_robot_coord(parts[1],board_coords)
                royal = True
                capture_piece(end_square, tool_orientation, royal)
                move_home(start_position, tool_orientation)
                move_piece(start_square, end_square, tool_orientation, royal)
                move_home(start_position, tool_orientation)

                """
                1. Send the setup to StockFish
                2. Do the move
                    i. Check for royal and check for capture
                    ii. Want to fix castle and pawn -> queen whatever it is called, This is called promotion
                3. Update the board? Or is that manual?
             
                """
        elif robot_mode == 3:
            
            # Ge boardstate till engine, som ger oss vad den tycker är det bästa movet
            propposed_move = best_move_local(current_fen, think_ms=500)
            print("The best move is ",propposed_move)

            #plocka ut vilka rutor det draget innebär (HÄR MÅSTE VI UNDERSÖKA OM MIN KOD FÖR CASTELING GER DET JAG FÖRVÄNTAR MIG ATT DEN SKA GE FÖR JAG HITTAR INTE DET PÅ INTERNET)
            start_square, end_square, promo = parse_uci_chess(propposed_move)

            #kolla först om movet den vill göra innebär en capture, om så, utför capturen
            if is_capture_move(current_fen,propposed_move):
                capture_piece(end_square,tool_orientation,is_royal_piece(current_fen,end_square))
                move_home(start_position, tool_orientation)
                move_piece(chess_coord_to_robot_coord(start_square,board_coords), chess_coord_to_robot_coord(end_square,board_coords), tool_orientation, is_royal_piece(current_fen,start_square))
                move_home(start_position, tool_orientation)
            else: 
                #om movet inte är en capture, kolla då om det är en casteling move, om ja, utför casteling proceedure genom att skicka två move kommand.
                is_castleing, king_move,rook_move,side,kind = is_casteling_move(current_fen,propposed_move)
                if(is_castleing):
                    king_start_square, king_end_square = parse_uci_chess(king_move)
                    king_start_square_position = chess_coord_to_robot_coord(king_start_square,board_coords)
                    king_end_square_position = chess_coord_to_robot_coord(king_end_square,board_coords)
                    
                    rook_start_square, rook_end_square = parse_uci_chess(king_move)
                    rook_start_square_position = chess_coord_to_robot_coord(rook_start_square,board_coords)
                    rook_end_squareposition = chess_coord_to_robot_coord(rook_end_square,board_coords)

                    move_piece(king_start_square_position, king_end_square_position, tool_orientation,True) # Vi vet att kungen är royal    

                    move_piece(rook_start_square_position, rook_end_squareposition, tool_orientation) 
                    move_home(start_position, tool_orientation)
                else:

                    move_piece(start_square, end_square, tool_orientation,is_royal_piece(fen,start_square))            
                    move_home(start_position, tool_orientation)      

                
            newFen, san = update_fen_with_uci(current_fen,propposed_move)
        else:

            logging.CRITICAL("Bad robot mode input. Quitting.")


    except KeyboardInterrupt:
        logging.info("Shutting down the chessbot.")

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()
