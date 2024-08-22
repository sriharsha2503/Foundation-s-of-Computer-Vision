'''import cv2
import numpy as np
def unsharp_masking(image_path, output_path, sigma=1.0, alpha=1.5, beta=0.0):
    image = cv2.imread("/home/student/220962019/opencv/lab3/image.jpeg")
    if image is None:
        raise ValueError("Image not found or unable to load.")
    image_float = image.astype(np.float32) / 255.0
    blurred_image = cv2.GaussianBlur(image_float, (0, 0), sigma)
    sharpened_image = cv2.addWeighted(image_float, alpha, blurred_image, beta, 0)
    sharpened_image = np.clip(sharpened_image, 0.0, 1.0)
    sharpened_image = (sharpened_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, sharpened_image)
    print(f"Sharpened image saved to {output_path}")
unsharp_masking('image.jpg', 'sharpened_image.jpg')
'''
import cv2
import numpy as np


def custom_blur(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    return cv2.filter2D(image, -1, kernel)


def unsharp_masking(image, kernel_size, alpha):
    blurred_image = custom_blur(image, kernel_size)

    image_float = image.astype(np.float32)
    blurred_image_float = blurred_image.astype(np.float32)

    mask = cv2.subtract(image_float, blurred_image_float)

    sharpened_image_float = cv2.add(image_float, alpha * mask)

    sharpened_image = np.clip(sharpened_image_float, 0, 255).astype(np.uint8)

    return sharpened_image


def main():
    image_path = 'image.jpeg'
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return

    kernel_size = 5
    alpha = 1.5


    sharpened_image = unsharp_masking(image, kernel_size, alpha)

    cv2.imshow('Original Image', image)
    cv2.imshow('Sharpened Image', sharpened_image)

    cv2.imwrite('sharpened_image2.jpeg', sharpened_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
