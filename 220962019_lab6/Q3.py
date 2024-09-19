import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    return keypoints

def extract_descriptors(image, keypoints, window_size=16):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    descriptors = []
    half_window = window_size // 2

    for kp in keypoints:
        x, y = kp
        if (x >= half_window and x < gray.shape[1] - half_window and
            y >= half_window and y < gray.shape[0] - half_window):
            patch = gray[y-half_window:y+half_window, x-half_window:x+half_window]
            patch_resized = cv2.resize(patch, (window_size, window_size))
            descriptors.append(patch_resized.flatten())

    return np.array(descriptors)

def match_descriptors(desc1, desc2):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = np.linalg.norm(desc2 - d1, axis=1)
        best_match = np.argmin(distances)
        matches.append((i, best_match))
    return matches

def draw_matches(img1, kp1, img2, kp2, matches, color=(255, 0, 0)):
    # Resize img2 to match the height of img1
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    if height1 != height2:
        img2_resized = cv2.resize(img2, (width2 * height1 // height2, height1))
    else:
        img2_resized = img2

    img3 = np.hstack((img1, img2_resized))

    for m in matches:
        i, j = m
        pt1 = tuple(kp1[i][::-1])
        pt2 = tuple(kp2[j][::-1] + np.array([img1.shape[1], 0]))  # Adjust x-coordinate for img2
        cv2.line(img3, pt1, pt2, color, 1)

    return img3

# Read images
img1 = cv2.imread('/home/student/220962019/opencv/lab6/i1.jpeg')
img2 = cv2.imread('/home/student/220962019/opencv/lab6/i2.jpeg')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Detect keypoints
kp1 = detect_keypoints(img1_rgb)
kp2 = detect_keypoints(img2_rgb)

# Extract descriptors
desc1 = extract_descriptors(img1_rgb, kp1)
desc2 = extract_descriptors(img2_rgb, kp2)

# Match descriptors
matches = match_descriptors(desc1, desc2)

# Draw matches
img_matches = draw_matches(img1_rgb, kp1, img2_rgb, kp2, matches)

# Display the results
plt.imshow(img_matches)
plt.axis('off')  # Hide axes
plt.show()
