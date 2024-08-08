import cv2
import numpy as np
def apply_filters(image):
    box_kernel_size = (5, 5)
    gaussian_kernel_size = (5, 5)
    box_kernel = np.ones(box_kernel_size) / (box_kernel_size[0] * box_kernel_size[1])
    box_filtered_image = cv2.filter2D(image, -1, box_kernel)
    gaussian_filtered_image = cv2.GaussianBlur(image, gaussian_kernel_size, 0)
    return box_filtered_image, gaussian_filtered_image


def save_images(original, box_filtered, gaussian_filtered):
    cv2.imwrite('/home/student/220962019/opencv/lab3/original_image.jpg', original)
    cv2.imwrite('/home/student/220962019/opencv/lab3/box_filtered_image.jpg', box_filtered)
    cv2.imwrite('/home/student/220962019/opencv/lab3/gaussian_filtered_image.jpg', gaussian_filtered)
def main():
    image_path = '/home/student/220962019/opencv/lab3/image.jpg'
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image. Check the path.")
        return
    box_filtered_image, gaussian_filtered_image = apply_filters(image)
    save_images(image, box_filtered_image, gaussian_filtered_image)
if __name__ == "__main__":
    main()
