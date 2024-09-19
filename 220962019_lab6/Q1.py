import cv2
import numpy as np

def detect_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(image, None)
    return kp, des

def match_descriptors(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches

def compute_homography_ransac(pts1, pts2, iterations=1000, threshold=3):
    if len(pts1) < 4:
        raise ValueError("Not enough points to compute homography")

    best_H = None
    max_inliers = 0
    n = len(pts1)

    for _ in range(iterations):
        indices = np.random.choice(n, 4, replace=False)
        src_pts = pts1[indices]
        dst_pts = pts2[indices]

        H = compute_homography(src_pts, dst_pts)

        if H is not None:
            inliers = compute_inliers(pts1, pts2, H, threshold)
            num_inliers = len(inliers)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H

    return best_H

def compute_homography(src_pts, dst_pts):
    A = []
    for i in range(4):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x*u, y*u, u])
        A.append([0, 0, 0, -x, -y, -1, x*v, y*v, v])
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    return H

def compute_inliers(src_pts, dst_pts, H, threshold):
    inliers = []
    for i in range(len(src_pts)):
        pt1 = np.append(src_pts[i], 1)
        pt2 = np.append(dst_pts[i], 1)
        projected_pt = np.dot(H, pt1)
        projected_pt /= projected_pt[2]
        distance = np.linalg.norm(projected_pt[:2] - pt2[:2])
        if distance < threshold:
            inliers.append(i)
    return inliers

def draw_matches(img1, kp1, img2, kp2, matches):
    # Convert keypoints to list of (x, y) tuples
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches

# Load images
img1 = cv2.imread('/home/student/220962019/opencv/lab6/i1.jpeg')
img2 = cv2.imread('/home/student/220962019/opencv/lab6/i2.jpeg')

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect keypoints and compute descriptors
kp1, des1 = detect_keypoints_and_descriptors(gray1)
kp2, des2 = detect_keypoints_and_descriptors(gray2)

if des1 is None or des2 is None:
    raise ValueError("Descriptors could not be computed")

# Match descriptors
matches = match_descriptors(des1, des2)

if not matches:
    raise ValueError("No matches found between descriptors")

# Extract matched keypoints
pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

if len(pts1) < 4:
    raise ValueError("Not enough matched points to compute homography")

# Compute homography using RANSAC
H = compute_homography_ransac(pts1, pts2)

print("Homography Matrix:")
print(H)

# Draw and display matches
img_matches = draw_matches(img1, kp1, img2, kp2, matches)

cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()

