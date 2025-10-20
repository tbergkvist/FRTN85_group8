from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
#from smc.control.cartesian_space.cartesian_space_compliant_control import compliantMoveL
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
import pinocchio as pin

import argparse
import numpy as np
import time


def dummy_streamer():
    # Emulate the computer vision stream.
    while True:
        time.sleep(0.5)
        yield {"class": [1], "confidence": [0.9], "center_point": [0.1, 0.1, 0.1], "BBox": [0.1, 0.1, 0.2, 0.2], "data": True, "color_frame": np.zeros((640, 360))}


def convert_coords(coords, from_file=False):
    # Measure these using the calibration_script.

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
    above = coords.copy() + np.array([0.01, 0, 0.25])
    on = coords.copy() + np.array([0.01, 0, 0.15])
    place = coords.copy() + np.array([0.01, 0, 0.155])
    return above, on, place


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
    args = parser.parse_args()
    return args

number2piece = {0: 'black bishop',
                1: 'black king',
                2: 'black knight',
                3: 'black pawn',
                4: 'black queen',
                5: 'black rook',
                6: 'white bishop',
                7: 'white king',
                8: 'white knight',
                9: 'white pawn',
                10: 'white queen',
                11: 'white rook'}

piece2number = {'black bishop': 0,
                'black king'  : 1,
                'black knight': 2,
                'black pawn'  : 3,
                'black queen' : 4,
                'black rook'  : 5,
                'white bishop': 6,
                'white king'  : 7,
                'white knight': 8,
                'white pawn'  : 9,
                'white queen': 1,
                'white rook' : 1}


if __name__ == "__main__":
    args = get_args()
    robot = getRobotFromArgs(args)
 
    print("Initializing realsense stream.")
    if args.realsense:
        from computer_vision import stream_camera_frame_coords # Need the realsense sdk for this import.
        realsense_stream = stream_camera_frame_coords(multiple_pieces=True, piece_conf_thres=0.4) 
    else:
        realsense_stream = dummy_streamer() 
    
    robot._step()
    initial_rotation = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])
    if(args.start_from_current_pose):
        initial_position = robot.T_w_e.translation
    else:
        if args.real:
            print("Use --start-from-current-pose")
            quit()
        initial_position = np.array([0.3, 0.3, 0.5]) # dont use this one, it is inside wall.
    
    print("Initial position of robot")    
    print(initial_position)

    print("Moving to initial pose.")
    T_w_goal = pin.SE3(initial_rotation, initial_position)
    moveL(args, robot, T_w_goal)

 
    while True:
        try:
            pieces = next(realsense_stream)
            print(f"Currently seeing these pieces: {[number2piece[p] for p in pieces["class"]]}")
            piece = None
            while piece is None:
                try:
                    piece = piece2number[input("Piece to move: ").lower()]
                except:
                    print("Bad input.")
            command = np.array([float(val) for val in input("Where to move piece: x.x,y.y: ").split(",")])
            command = np.append(command, 0)
            print("Will move the piece this much in x and y: ", command)

            print("Looking for a chess piece using realsense camera.")
            piece_coords = None
            for i in range(5):
                pieces = next(realsense_stream)
                print("Pieces", pieces)
                if piece in pieces["class"]:
                    index = pieces["class"].index(piece)
                    piece_coords = pieces["center_point"][index]
                    piece_coords = convert_coords(piece_coords, "./H.txt")
                    print("Chess piece found at: ", piece_coords)
                    break
            if piece_coords is None:
                print("Could not find your piece. Try again.")
                continue
            above, on, place = get_grip_positions(piece_coords)
            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            robot.openGripper()
            print("Has moved to position above the piece: ", above)
            
            T_w_goal = pin.SE3(initial_rotation, on)
            moveL(args, robot, T_w_goal)

            robot.closeGripper()
            time.sleep(1)
            print("Has moved to position on the piece and closed gripper: ", on)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            print("Has lifted the piece to", above)

            new_pos = piece_coords + command
            above, on, place = get_grip_positions(new_pos)
            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            print("Has moved the piece to above new position: ", above)

            T_w_goal = pin.SE3(initial_rotation, place)
            moveL(args, robot, T_w_goal)
            robot.openGripper()
            time.sleep(1)
            print("Has put down the piece at: ", place)

            T_w_goal = pin.SE3(initial_rotation, initial_position)
            moveL(args, robot, T_w_goal)
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
