from computer_vision import stream_camera_frame_coords
import numpy as np

realsense_stream = stream_camera_frame_coords()

points_rs = []
points_robot = []

while True:
    try:
        print(f"Collect {len(points_rs)}st point.")
        input("Enter to collect point from realsense.")
        points_rs.append(next(realsense_stream)) # Adds np.array([x, y, z])
        print("Move robot to the chess piece position.")
        robot_point = np.array([float(val) for val in input("Enter the robot coordinates.").split(",")])
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

# Compute covariance matrix
H = rs_centered.T @ robot_centered

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(H)

# Compute rotation
R = Vt.T @ U.T

# Handle reflection (determinant negative)
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T

# Compute translation
t = centroid_robot - R @ centroid_rs

# Compose homogeneous transformation matrix
H = np.eye(4)
H[:3, :3] = R
H[:3, 3] = t

print("\nHomogeneous transformation (from RealSense to robot frame):")
print(H)
H.tofile("./")
