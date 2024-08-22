import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(title, img, cmap='gray'):
    """Displays an image with a given title using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


def segment_color(image_path, lower_bound, upper_bound):
    """Segments the image based on the specified color range."""
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    display_image('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    display_image('Mask', mask, cmap='gray')
    display_image('Segmented Image', cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))


path = '/home/student/220962019/opencv/lab4/Lenna.png'
lb = [150, 100, 100]  # Lower bound of HSV for red color
ub = [200, 255, 255]  # Upper bound of HSV for red color
segment_color(path, lb, ub)
