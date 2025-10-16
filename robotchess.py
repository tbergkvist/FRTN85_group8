from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
from smc.control.cartesian_space.cartesian_space_compliant_control import compliantMoveL
import pinocchio as pin

import argparse
import numpy as np
import time

########## Note from Teo ###############
### This is some real spaghetti code ###
########################################


def dummy_streamer():
    # simlutate the computer vision stream.
    while True:
        time.sleep(0.5)
        yield 0.3, -0.3, 0.05

def convert_coords(coords):
    x, y, z = coords

    #R = np.array([[-0.021493, -0.41323, 0.91037], # MEASURE THIS ONE MANUALLY.
     #           [-0.99707, 0.075723, 0.010832],
      #          [-0.073413, -0.90747, -0.41365 ]])
    R = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])

    #t = np.array([324.42, -463.97, 10.465]) # MEASURE THIS ONE MANUALLY.
    t = np.array([0, 0, 0])

    H = np.eye(4, dtype=float)
    H[:3, :3] = R
    H[:3,  3] = t
    return (H @ np.array([x, y, z, 1]))[:3]

def get_grip_positions(coords):
    above = coords.copy() + np.array([0, 0, 0.25])
    on = coords.copy() + np.array([0, 0, 0.15])
    return above, on

def zero_robot_vel():
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    robot = getRobotFromArgs(args)

    print("Initializing realsense stream.")
    if args.realsense:
        from computer_vision import stream_camera_frame_coords #need the realsense software for this import 
        realsense_stream = stream_camera_frame_coords() 
    else:
        realsense_stream = dummy_streamer() 
    
    robot._step()
    initial_rotation = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])
    if(args.start_from_current_pose):
        initial_position = robot.T_w_e.translation
    else:   
        initial_position = np.array([0.3, 0.3, 0.5])
    
    print("Initial position of robot")    
    print(initial_position)

    print("Moving to initial pose.")
    T_w_goal = pin.SE3(initial_rotation, initial_position)
    #moveL(args, robot, T_w_goal)
    compliantMoveL(T_w_goal, args, robot)
    zero_robot_vel()
 
    while True:
        try:
            command = np.array([float(val) for val in input("Where to move piece: x.x,y.y: ").split(",")])
            command = np.append(command, 0)
            print("Will move the piece this much in x and y: ", command)

            print("Looking for a chess piece using realsense camera.")
            piece_coords = next(realsense_stream)
            piece_coords = convert_coords(piece_coords)
            print("Chess piece found at: ", piece_coords)

            above, on = get_grip_positions(piece_coords)
            T_w_goal = pin.SE3(initial_rotation, above)
            #moveL(args, robot, T_w_goal)
            compliantMoveL(T_w_goal, args, robot)
            robot.openGripper()
            print("Has moved to position above the piece: ", above)
            
            T_w_goal = pin.SE3(initial_rotation, on)
            #moveL(args, robot, T_w_goal)
            compliantMoveL(T_w_goal, args, robot)
            zero_robot_vel()
            robot.closeGripper()
            time.sleep(1)
            print("Has moved to position on the piece and closed gripper: ", on)

            T_w_goal = pin.SE3(initial_rotation, above)
            #moveL(args, robot, T_w_goal)
            compliantMoveL(T_w_goal, args, robot)
            print("Has lifted the piece to", above)
            

            new_pos = piece_coords + command
            above, on = get_grip_positions(new_pos)
            T_w_goal = pin.SE3(initial_rotation, above)
            #moveL(args, robot, T_w_goal)
            compliantMoveL(T_w_goal, args, robot)
            print("Has moved the piece to above new position: ", above)

            T_w_goal = pin.SE3(initial_rotation, on)
            #moveL(args, robot, T_w_goal)
            compliantMoveL(T_w_goal, args, robot)
            zero_robot_vel()
            robot.openGripper()
            time.sleep(1)
            print("Has put down the piece at: ", on)

            T_w_goal = pin.SE3(initial_rotation, initial_position)
            #moveL(args, robot, T_w_goal) 
            compliantMoveL(T_w_goal, args, robot)
            zero_robot_vel()
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
