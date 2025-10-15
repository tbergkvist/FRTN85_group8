from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
from smc.robots.utils import defineGoalPointCLI
from smc.control.cartesian_space.cartesian_space_point_to_point import moveL
import pinocchio as pin

import argparse
import numpy as np
import time

# This script just makes the robot go to some different points, and for each point it does some rotations.
# Over and over again.


def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Chess playing robot madness."
    parser = getClikArgs(parser)
    args = parser.parse_args()
    return args


t0 = np.array([0.0, 0.0, 0.0])
t1 = np.array([0.3, 0.3, 0.3])
t2 = np.array([-0.3, 0.3, 0.3])
t3 = np.array([0.3, -0.3, 0.3])

R0 = np.eye(3)

R1 = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])

R2 = np.array([
    [np.sqrt(2)/2, 0,  np.sqrt(2)/2],
    [0, 1, 0],
    [-np.sqrt(2)/2, 0, np.sqrt(2)/2]
])

R3 = np.array([
    [-0.5, -np.sqrt(3)/2, 0],
    [ np.sqrt(3)/2, -0.5, 0],
    [0, 0, 1]
])

t_all = [t1, t2, t3]
R_all = [R1, R2, R3]


if __name__ == "__main__":
    args = get_args()
    robot = getRobotFromArgs(args)

    while True:
        T_w_goal = pin.SE3(R0, t0)
        time.sleep(3)
        try:
            for t in t_all:
                for R in R_all:
                    T_w_goal = pin.SE3(R, t)
                    moveL(args, robot, T_w_goal)
                    print(robot.model)
                    time.sleep(3)

        except KeyboardInterrupt:
            print("Breaking loop, please wait.")
            time.sleep(1)
            break

    if args.real:
        robot.stopRobot()

    if args.visualizer:
        robot.killManipulatorVisualizer()

    if args.save_log:
        robot._log_manager.saveLog()
