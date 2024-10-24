import cv2
import numpy as np
import glob

CHECKERBOARD = (12, 12)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)

objpoints = []
imgpoints = []

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

images = glob.glob('./calib_example/*.tif')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix (Intrinsic parameters): \n", mtx)
print("Distortion coefficients: \n", dist)

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for rvec, tvec in zip(rvecs, tvecs):
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)

        for point in imgpoints2:
            cv2.circle(img, tuple(point.ravel().astype(int)), 5, (0, 255, 0), -1)

    cv2.imshow('Reprojected Points', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
