from computer_vision import stream_camera_frame_coords
import numpy as np

from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
import pinocchio as pin
import argparse

def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Chess playing robot madness."
    parser = getClikArgs(parser)
    parser.add_argument(
        "--manual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, add the robot points manually as command line input.",
    )
    args = parser.parse_args()
    return args



args = get_args()
robot = getRobotFromArgs(args)

robot._step()

realsense_stream = stream_camera_frame_coords()

points_rs = []
points_robot = []

while True:
    try:
        print(f"Collect {len(points_rs)}st point.")
        input("Enter to collect point from realsense.")
        points_rs.append(next(realsense_stream))
        print("Move robot to the chess piece position.")
        if args.manual:
            robot_point = np.array([float(val) for val in input("Enter the robot coordinates.").split(",")])
        else:
            input("Measure robot point.")
            robot_point = robot.T_w_e.translation
        points_robot.append(robot_point)
        print("Saved pair. Continue to save at least 5 points.")
    except KeyboardInterrupt:
        break

points_rs = np.array(points_rs)
points_robot = np.array(points_robot)

if len(points_rs) < 3:
    raise ValueError("At least 3 non-collinear points are required to compute a transformation.")

# Compute centroids
centroid_rs = np.mean(points_rs, axis=0)
centroid_robot = np.mean(points_robot, axis=0)

# Center the points
rs_centered = points_rs - centroid_rs
robot_centered = points_robot - centroid_robot

# Compute covariance matrix (robot ← rs)
H = robot_centered.T @ rs_centered

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(H)

# Compute rotation
R = U @ Vt

# Handle possible reflection
if np.linalg.det(R) < 0:
    U[:, -1] *= -1
    R = U @ Vt

# Compute translation
t = centroid_robot - R @ centroid_rs


# Compose homogeneous transformation matrix
H = np.eye(4)
H[:3, :3] = R
H[:3, 3] = t

print("\nHomogeneous transformation (from RealSense to robot frame):")
print(H)
H.tofile("./H.txt")

if len(points_rs) > 3:
    test_index = -1  # Use the 4th point (index 3) as test point

    # Get the test points
    p_rs_test = np.append(points_rs[test_index], 1)  # Homogeneous form
    p_robot_actual = points_robot[test_index]

    # Predicted point in robot frame
    p_robot_pred = (H @ p_rs_test)[:3]

    # Compute Euclidean error
    error = np.linalg.norm(p_robot_pred - p_robot_actual)

    print(f"\nCalibration check using point {test_index + 1}:")
    print(f"Predicted (robot frame): {p_robot_pred}")
    print(f"Actual (robot frame):    {p_robot_actual}")
    print(f"Error (m): {error:.6f}")
else:
    print("\nNot enough points for calibration error check (need at least 4).")
