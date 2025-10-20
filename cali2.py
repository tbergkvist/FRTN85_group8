import numpy as np
from smc import getMinimalArgParser, getRobotFromArgs
from smc.control.cartesian_space import getClikArgs
import pinocchio as pin
import argparse
import cv2
import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import pyrealsense2 as rs
from collections import deque


def get_args() -> argparse.Namespace:
    parser = getMinimalArgParser()
    parser.description = "Chess playing robot calibration."
    parser = getClikArgs(parser)
    parser.add_argument(
        "--manual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If True, enter robot points manually as comma separated values x,y,z.",
    )
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=5,
        help="Collect at least this many point pairs before solving.",
    )
    parser.add_argument(
        "--depth_win",
        type=int,
        default=5,
        help="Odd window size for neighborhood depth search when the center depth is invalid.",
    )
    return parser.parse_args()


def enable_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable both streams; resolution and FPS can be adjusted if needed
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    # Align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Color intrinsics for deprojection
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = color_stream.get_intrinsics()
    return pipeline, align, color_intr


class ClickGUI:
    def __init__(self, window_name="RealSense Color"):
        self.window = window_name
        self.click = None  # (x, y)
        self.history = deque(maxlen=50)
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click = (x, y)
            self.history.append((x, y))

    def draw_overlay(self, frame):
        if self.click is not None:
            x, y = self.click
            cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)
            cv2.putText(frame, f"Click: ({x}, {y})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Controls: click to select, 'c'=capture, 'r'=reset, 'q'=finish",
                    (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    def reset(self):
        self.click = None

    def get_click(self):
        return self.click


def depth_at_pixel_or_neighborhood(depth_frame, x, y, win=5):
    """Return a metric depth in meters at integer pixel (x, y).
    If center is invalid, search a win×win neighborhood and return median of valid."""
    Z = depth_frame.get_distance(int(x), int(y))
    if Z and Z > 0.0:
        return float(Z)
    # Neighborhood search
    if win < 1 or win % 2 == 0:
        win = 5
    half = win // 2
    vals = []
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    for j in range(max(0, y - half), min(h, y + half + 1)):
        for i in range(max(0, x - half), min(w, x + half + 1)):
            z = depth_frame.get_distance(int(i), int(j))
            if z and z > 0.0:
                vals.append(float(z))
    if not vals:
        return None
    return float(np.median(vals))


def solve_rigid_transform(A, B):
    """Compute rigid transform that maps A to B: B ≈ R A + t.
    A and B are shaped (N, 3). Returns 4×4 homogeneous matrix H."""
    if A.shape[0] < 3 or B.shape[0] < 3:
        raise ValueError("At least 3 non collinear points are required.")
    if A.shape != B.shape:
        raise ValueError("Point sets must have the same shape.")

    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    A0 = A - centroid_A
    B0 = B - centroid_B

    H = B0.T @ A0
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = centroid_B - R @ centroid_A

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def main():
    args = get_args()
    robot = getRobotFromArgs(args)
    robot._step()
    robot.setFreedrive()

    pipeline, align, color_intr = enable_realsense()

    # Storage
    points_rs = []
    points_robot = []

    gui = ClickGUI("RealSense Color")

    print("Instructions:")
    print(f"- Click a pixel to select \x28x, y\x29, press 'c' to capture the RealSense point.")
    print(f"- Press 'q' anytime to finish and solve once you have at least {args.min_pairs} pairs.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)

            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())

            # Draw overlay
            vis = color.copy()
            gui.draw_overlay(vis)
            cv2.imshow(gui.window, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                gui.reset()
            elif key == ord('c'):
                click = gui.get_click()
                if click is None:
                    print("No click selected. Please click on the image first.")
                    continue
                x, y = int(click[0]), int(click[1])
                Z = depth_at_pixel_or_neighborhood(depth_frame, x, y, win=args.depth_win)
                if Z is None or Z <= 0.0:
                    print("Invalid depth at selection. Try another pixel.")
                    continue

                # Deproject uses float pixel coordinates
                X, Y, Zm = rs.rs2_deproject_pixel_to_point(color_intr, [float(x), float(y)], float(Z))
                pts_rs = np.array([X, Y, Zm], dtype=float)
                points_rs.append(pts_rs)
                print(f"RealSense point {len(points_rs)}: {pts_rs}")

                if args.manual:
                    robot_point = np.array(
                        [float(val) for val in input("Enter robot coordinates x,y,z in meters: ").split(",")],
                        dtype=float
                    )
                else:
                    robot._step()
                    # Get homogeneous 4x4 for flange pose
                    T_w_e = np.array(robot.T_w_e)  # ensure this is 4x4; if it's an SE3, use .homogeneous
                    print(f"T_w_e is currently: {T_w_e}")

                    L = 0.12  # meters (12 cm)
                    p_tool_e = np.array([0.0, 0.0, L, 1.0])  # tool tip in end-effector frame

                    # Tool tip in world frame
                    p_tool_w = (T_w_e @ p_tool_e)[:3]  # take only x,y,z

                    robot_point = p_tool_w

                points_robot.append(robot_point)
                print(f"Robot point {len(points_robot)}: {robot_point}")
                print(f"Saved pair {len(points_rs)}. Collect at least {args.min_pairs} pairs.")

            elif key == ord('q'):
                if len(points_rs) < max(3, args.min_pairs):
                    print(f"Need at least {max(3, args.min_pairs)} pairs to solve. Currently: {len(points_rs)}")
                    continue
                print("Finishing collection.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    points_rs = np.array(points_rs, dtype=float)
    points_robot = np.array(points_robot, dtype=float)

    if len(points_rs) < 3:
        raise ValueError("At least 3 non collinear points are required to compute a transformation.")

    # Solve RealSense to robot transform
    H = solve_rigid_transform(points_rs, points_robot)

    print("\nHomogeneous transformation \x28from RealSense to robot frame\x29:")
    np.set_printoptions(precision=6, suppress=True)
    print(H)

    # Persist results
    H.tofile("./H.txt")
    print('Saved H to "./H.txt".')

    # Quick hold out test if we have 4 or more
    if len(points_rs) > 3:
        test_index = -1
        p_rs_test = np.append(points_rs[test_index], 1.0)
        p_robot_actual = points_robot[test_index]
        p_robot_pred = (H @ p_rs_test)[:3]
        error = np.linalg.norm(p_robot_pred - p_robot_actual)
        print(f"\nCalibration check using point {test_index + 1}:")
        print(f"Predicted \x28robot frame\x29: {p_robot_pred}")
        print(f"Actual    \x28robot frame\x29: {p_robot_actual}")
        print(f"Error \x28m\x29: {error:.6f}")
    else:
        print("\nNot enough points for calibration error check \x28need at least 4\x29.")


if __name__ == "__main__":
    main()
