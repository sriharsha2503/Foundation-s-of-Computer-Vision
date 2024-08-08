import cv2
import numpy as np
def compute_image_gradient(image_path, output_path_x, output_path_y, output_path_magnitude):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    grad_x = cv2.normalize(grad_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    grad_y = cv2.normalize(grad_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(output_path_x, grad_x)
    cv2.imwrite(output_path_y, grad_y)
    cv2.imwrite(output_path_magnitude, magnitude)

    print(f"Gradient images saved to {output_path_x}, {output_path_y}, and {output_path_magnitude}")

compute_image_gradient(
    "/home/student/220962019/opencv/lab3/image.jpeg",
    "/home/student/220962019/opencv/lab3/gradient_x.jpg",
    "/home/student/220962019/opencv/lab3/gradient_y.jpg",
    "/home/student/220962019/opencv/lab3/gradient_magnitude.jpg"
)
