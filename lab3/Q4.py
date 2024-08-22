'''import cv2
def detect_edges(image_path, output_path, lower_threshold=50, upper_threshold=150):

    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image. Check the path.")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, lower_threshold, upper_threshold)

    cv2.imwrite(output_path, edges)

    print(f"Edge detection complete. Output saved to {output_path}")


def main():

    input_image_path = '/home/student/220962019/opencv/lab3/image.jpeg'
    output_image_path = '/home/student/220962019/opencv/lab3/edges_detected.jpg'

    detect_edges(input_image_path, output_image_path)


if __name__ == "__main__":
    main()
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


def laplacian_edge_detection(image):
    # Define the Laplacian kernel
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]], dtype=np.float64)

    laplacian_edges = apply_kernel(image, laplacian_kernel)

    # Normalize the result to the range [0, 255]
    laplacian_edges = np.abs(laplacian_edges)
    normalized_edges = cv2.normalize(laplacian_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return normalized_edges


def main():
    image_path = 'image.jpeg'  # Change this to your image path
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print("Error: Image not found.")
        return

    edge_magnitude = laplacian_edge_detection(image)

    cv2.imshow('Original Image', image)
    cv2.imshow('Edge Detection', edge_magnitude)

    cv2.imwrite('laplacian_edge_detection2.jpg', edge_magnitude)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
