import cv2
feather=cv2.imread("./images/img1.jpeg",0)
cv2.imshow("holy god",feather)
cv2.imwrite("./images/img2.jpeg",feather)
cv2.waitKey(0)
