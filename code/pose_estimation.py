import numpy as np
import cv2 as cv

# The given video and calibration data
video_file = "data/chessboard.avi"
K = np.array(
    [
        [607.21421351, 0.0, 631.61705825],
        [0.0, 611.4314851, 376.7056395],
        [0.0, 0.0, 1.0],
    ]
)
dist_coeff = np.array([0.01249517, -0.00812537, -0.00134196, -0.00195656, 0.00709386])
board_pattern = (8, 6)
board_cellsize = 0.025
board_criteria = (
    cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
)

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), "Cannot read the given input, " + video_file

# make a video
frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*"XVID")
out = cv.VideoWriter("data/output.avi", fourcc, fps, (frame_width, frame_height))

# Prepare Coment "CV" Block
# C
boxC_front = board_cellsize * np.array(
    [
        [0, 2, -3],
        [3, 2, -3],
        [3, 2, -2],
        [1, 2, -2],
        [1, 2, -1],
        [3, 2, -1],
        [3, 2, 0],
        [0, 2, 0],
    ]
)
boxC_back = board_cellsize * np.array(
    [
        [0, 3, -3],
        [3, 3, -3],
        [3, 3, -2],
        [1, 3, -2],
        [1, 3, -1],
        [3, 3, -1],
        [3, 3, 0],
        [0, 3, 0],
    ]
)
# V
boxV_front = board_cellsize * np.array(
    [
        [4, 2, -3],
        [5, 2, -3],
        [5.5, 2, -1],
        [6, 2, -3],
        [7, 2, -3],
        [6, 2, 0],
        [5, 2, 0],
    ]
)
boxV_back = board_cellsize * np.array(
    [
        [4, 3, -3],
        [5, 3, -3],
        [5.5, 3, -1],
        [6, 3, -3],
        [7, 3, -3],
        [6, 3, 0],
        [5, 3, 0],
    ]
)

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
)

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw the C on the image
        line_C_front, _ = cv.projectPoints(boxC_front, rvec, tvec, K, dist_coeff)
        line_C_back, _ = cv.projectPoints(boxC_back, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_C_front)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_C_back)], True, (0, 0, 255), 2)
        for b, t in zip(line_C_front, line_C_back):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Draw the V on the image
        line_V_front, _ = cv.projectPoints(boxV_front, rvec, tvec, K, dist_coeff)
        line_V_back, _ = cv.projectPoints(boxV_back, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(line_V_front)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(line_V_back)], True, (0, 0, 255), 2)
        for b, t in zip(line_V_front, line_V_back):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (0, 255, 0), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)  # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f"XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]"
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    out.write(img)

    # Show the image and process the key event
    cv.imshow("Pose Estimation (Chessboard)", img)
    key = cv.waitKey(10)
    if key == ord(" "):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
out.release()
cv.destroyAllWindows()
