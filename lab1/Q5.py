import cv2
feather=cv2.imread("./images/img1.jpeg")
ans=cv2.resize(feather,(1080,1080))
cv2.imwrite("./images/img3.jpeg",ans)
cv2.waitKey(0)