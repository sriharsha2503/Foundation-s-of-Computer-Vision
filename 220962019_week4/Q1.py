import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    return image

def global_thresholding(image, threshold_value):
    # Apply global thresholding
    binary_image = np.where(image >= threshold_value, 255, 0).astype(np.uint8)
    return binary_image

def adaptive_thresholding(image, block_size, C):
    # Initialize the output binary image
    binary_image = np.zeros_like(image, dtype=np.uint8)
    rows, cols = image.shape

    # Pad the image to handle borders
    padded_image = np.pad(image, pad_width=((block_size//2, block_size//2), (block_size//2, block_size//2)), mode='constant', constant_values=0)

    for i in range(rows):
        for j in range(cols):
            # Extract the local block
            local_block = padded_image[i:i+block_size, j:j+block_size]
            # Calculate the local threshold
            local_mean = np.mean(local_block)
            local_threshold = local_mean - C
            # Apply thresholding
            binary_image[i, j] = 255 if image[i, j] >= local_threshold else 0

    return binary_image

def otsu_thresholding(image):
    # Compute the histogram and its cumulative distribution
    hist, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size
    total_sum = np.sum(np.arange(256) * hist)

    # Initialize variables for Otsu's method
    sumB = 0
    weightB = 0
    maximum_variance = 0
    threshold = 0

    for i in range(256):
        weightF = np.sum(hist[i:])
        if weightF == 0:
            break

        weightB += hist[i]
        if weightB == 0:
            continue

        weightF -= hist[i]
        if weightF == 0:
            break

        sumB += i * hist[i]
        meanB = sumB / weightB
        meanF = (total_sum - sumB) / weightF
        between_class_variance = weightB * weightF * (meanB - meanF) ** 2

        if between_class_variance > maximum_variance:
            maximum_variance = between_class_variance
            threshold = i

    # Apply Otsu's thresholding
    binary_image = np.where(image >= threshold, 255, 0).astype(np.uint8)
    return binary_image

def display_images(images, titles):
    # Display images using matplotlib
    plt.figure(figsize=(12, 8))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 2, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path to your image
    image_path = '/home/student/220962019/opencv/lab4/Lenna.png'

    # Load image
    image = load_image(image_path)

    # Apply thresholding methods
    global_thresh = global_thresholding(image, threshold_value=127)  # Example threshold value
    adaptive_thresh = adaptive_thresholding(image, block_size=11, C=2)  # Example block size and C
    otsu_thresh = otsu_thresholding(image)

    # Display results
    images = [image, global_thresh, adaptive_thresh, otsu_thresh]
    titles = ['Original Image', 'Global Thresholding', 'Adaptive Thresholding', 'Otsu\'s Thresholding']
    display_images(images, titles)
