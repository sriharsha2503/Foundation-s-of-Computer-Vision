import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load images and convert to grayscale
img1 = cv.imread('book1.jpeg')
img2 = cv.imread('book2.jpeg')
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img1 = cv.resize(img1, (450, 300))

# Initialize SIFT detector
sift = cv.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Draw matches
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Show the matches
plt.imshow(img3), plt.title('Good Matches'), plt.show()

# Estimate homography using the good matches
if len(good) >= 4:  # At least 4 matches are needed to compute homography
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)

    # Get the corners of the first image
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    print(homography)
    # Transform corners to the second image
    dst = cv.perspectiveTransform(pts, homography)

    # Draw the polygon on the second image
    img2_with_rectangle = cv.polylines(img2, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)

    # Show the result
    plt.imshow(img2_with_rectangle, cmap='gray'), plt.title('Detected Homography'), plt.show()
else:
    print("Not enough good matches to estimate homography.")
