import cv2
img=cv2.imread("./images/img1.jpeg")
cv2.imshow("rotated Img",cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE))
cv2.waitKey(0)
