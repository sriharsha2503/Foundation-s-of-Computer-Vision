import cv2
import matplotlib.pyplot as plt
from cv2 import SIFT_create
# read images
img1 = cv2.imread(r'/home/student/220962019/opencv/lab6/i1.jpeg')
img2 = cv2.imread(r'/home/student/220962019/opencv/lab6/i2.jpeg')
img2 = cv2.resize(img2, (750, 1200))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
# sift
sift = SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
# feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)
line_color = (255, 0, 0)
img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:150],
img2, matchColor=line_color, singlePointColor=line_color, flags=2)
plt.imshow(img3),
plt.show()
