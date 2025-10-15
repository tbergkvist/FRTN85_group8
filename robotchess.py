from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
import pinocchio as pin

import argparse
import numpy as np
import time

#from computer_vision import stream_camera_frame_coords # uncomment when actually running the realsense.

########## Note from Teo ###############
### This is some real spaghetti code ###
########################################


def dummy_streamer():
    # simlutate the computer vision stream.
    while True:
        time.sleep(0.5)
        yield 0.3, 0.3, 0.05

def convert_coords(coords):
    x, y, z = coords
    R = np.array([[1, 0, 0], # MEASURE THIS ONE MANUALLY.
                [0, 1, 0],
                [0, 0, 1]])

    t = np.array([0, 0, 0]) # MEASURE THIS ONE MANUALLY.

    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    H[:3,  3] = t
    return (H @ np.array([x, y, z, 1]))[:3]

def get_grip_positions(coords):
    above = piece_coords.copy() + np.array([0, 0, 0.1])
    on = piece_coords.copy() + np.array([0, 0, 0.03])
    return above, on

def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Chess playing robot madness."
    parser = getClikArgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    robot = getRobotFromArgs(args)

    print("Initializing realsense stream.")
    realsense_stream = dummy_streamer() # stream_camera_frame_coords() # uncomment when actually running the realsense.


    initial_rotation = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, 1]])

    initial_position = np.array([0.3, 0.3, 0.5])


    print("Moving to initial pose.")
    T_w_goal = pin.SE3(initial_rotation, initial_position)
    moveL(args, robot, T_w_goal)

    while True:
        try:
            command = np.array([float(val) for val in input("Where to move piece: x.x,y.y").split(",")])
            command = np.append(command, 1)

            print("Looking for a chess piece using realsense camera.")
            piece_coords = next(realsense_stream)
            piece_coords = convert_coords(piece_coords)
            print("Chess piece found at: ", piece_coords, "\nConverting coords to robot frame.")

            above, on = get_grip_positions(piece_coords)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)
            robot.openGripper()
            
            T_w_goal = pin.SE3(initial_rotation, on)
            moveL(args, robot, T_w_goal)
            time.sleep(1)
            robot.closeGripper()
            time.sleep(1)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)

            new_pos = piece_coords[0] + command
            above, on = get_grip_positions(new_pos)

            T_w_goal = pin.SE3(initial_rotation, above)
            moveL(args, robot, T_w_goal)

            T_w_goal = pin.SE3(initial_rotation, on)
            moveL(args, robot, T_w_goal)
            robot.openGripper()

        except KeyboardInterrupt:
            print("Shutting down the chessbot.")
            break
        

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()
