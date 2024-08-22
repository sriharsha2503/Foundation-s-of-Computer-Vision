import cv2
import numpy as np

def apply_kmeans(image_path, k=8, output_path='kmeans_output.jpg'):

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    clustered_image = center[label.flatten()]
    clustered_image = clustered_image.reshape(image.shape)

    # Save the result to a file
    cv2.imwrite(output_path, clustered_image)


    cv2.imshow('Original Image', image)
    cv2.imshow('K-means Clustered Image', clustered_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_path = '/home/student/220962019/opencv/lab4/Lenna.png'  # Replace with your image path
output_path = 'kmeans_output.jpg'

apply_kmeans(image_path, k=8, output_path=output_path)
