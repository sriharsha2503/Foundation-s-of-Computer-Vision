import cv2
import numpy as np

def ratio_test(matches, ratio_threshold=0.75):
    reliable_matches = []

    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            reliable_matches.append(m)

    return reliable_matches


img1 = cv2.imread('/home/student/220962019/opencv/220962019_week6(2ndpart)/i1.jpeg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/student/220962019/opencv/220962019_week6(2ndpart)/i2.jpeg', cv2.IMREAD_GRAYSCALE)


sift = cv2.SIFT_create()


kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)


FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)


matches = flann.knnMatch(des1, des2, k=2)


reliable_matches = ratio_test(matches)


matched_img = cv2.drawMatches(img1, kp1, img2, kp2, reliable_matches, None)

cv2.imshow('Matched Keypoints', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
