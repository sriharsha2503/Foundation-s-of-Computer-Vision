'''import cv2
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
'''
import cv2
import numpy as np


def apply_kernel(image, kernel):
    rows, cols = image.shape
    ksize = kernel.shape[0]
    pad = ksize // 2

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(rows):
        for j in range(cols):
            region = padded_image[i:i + ksize, j:j + ksize]
            output[i, j] = np.sum(region * kernel)

    return output


def compute_gradients(image):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float64)

    grad_x = apply_kernel(image, sobel_x)
    grad_y = apply_kernel(image, sobel_y)

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)

    return grad_x, grad_y, magnitude, direction


def main():
    image_path = 'image.jpeg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    grad_x, grad_y, magnitude, direction = compute_gradients(image)

    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    direction = cv2.normalize(direction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    cv2.imshow('Gradient X', grad_x.astype(np.uint8))
    cv2.imshow('Gradient Y', grad_y.astype(np.uint8))
    cv2.imshow('Gradient Magnitude', magnitude)
    cv2.imshow('Gradient Direction', direction)

    cv2.imwrite('gradient_x2.jpg', grad_x.astype(np.uint8))
    cv2.imwrite('gradient_y2.jpg', grad_y.astype(np.uint8))
    cv2.imwrite('gradient_magnitude2.jpg', magnitude)
    cv2.imwrite('gradient_direction2.jpg', direction)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
