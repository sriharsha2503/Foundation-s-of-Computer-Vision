import cv2
import numpy as np
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (12, 12)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

# Creating vectors to store vectors of 3D points and 2D points for each checkerboard image
objpoints = []
imgpoints = []

# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Extracting paths of individual images stored in a given directory
images = glob.glob('./calib_example/*.tif')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Error reading image: {fname}")
        continue  # Skip this iteration if the image cannot be read

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)

        # Refining pixel coordinates for given 2D points
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)  # Adjust wait time for better visualization

cv2.destroyAllWindows()

# Performing camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Output intrinsic parameters
print("Camera Matrix (Intrinsic Parameters): \n", mtx)
print("Distortion Coefficients: \n", dist)

# Output extrinsic parameters for each image
for i in range(len(rvecs)):
    print(f"\nExtrinsic Parameters for image {i+1}:")
    print("Rotation Vector (rvec): \n", rvecs[i])
    print("Translation Vector (tvec): \n", tvecs[i])

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvecs[i])

    # Combine R and t into a projection matrix P
    RT = np.hstack((R, tvecs[i].reshape(-1, 1)))  # Combine R and t
    P = mtx @ RT  # Compute projection matrix
    print("Projection Matrix P: \n", P)
