import cv2
import numpy as np
import os
import glob

# Defining the dimensions of checkerboard
CHECKERBOARD = (12, 12)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

# Creating vectors to store 3D points and 2D image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Defining the world coordinates for 3D points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Extracting path of individual images stored in a given directory
images = glob.glob('./calib_example/*.tif')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If successful, append object points and image points
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)  # Display each image for a short time

cv2.destroyAllWindows()

# Performing camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Print intrinsic parameters
print("Camera matrix (Intrinsic parameters): \n", mtx)
print("Distortion coefficients: \n", dist)

# Print extrinsic parameters for each image
for i in range(len(rvecs)):
    print(f"Rotation vector for image {i}: \n", rvecs[i])
    print(f"Translation vector for image {i}: \n", tvecs[i])

# Optional: Save the calibration parameters to a file
np.savez('camera_calibration_parameters.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Reproject the points onto the images to verify calibration
for i in range(len(rvecs)):  # Loop based on the number of calibration results
    img = cv2.imread(images[i])  # Read the corresponding image
    h, w = img.shape[:2]

    # Reproject the corners
    imgpoints2, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], mtx, dist)

    # Draw the reprojected points
    for point in imgpoints2:
        cv2.circle(img, tuple(point[0].astype(int)), 5, (0, 255, 0), -1)

    # Display the image with reprojected points
    cv2.imshow(f'Reprojected Points - Image {i}', img)
    cv2.waitKey(500)  # Display each image for a short time

cv2.destroyAllWindows() 
