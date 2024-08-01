import numpy as np
import cv2


img=np.zeros((512,512,3))
cv2.rectangle(img,(100,150),(412,366),(0,0,255),3)
cv2.imshow("mordern art",img)
cv2.waitKey(0)