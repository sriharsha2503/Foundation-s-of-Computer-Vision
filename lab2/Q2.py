import cv2 as cv
import numpy as np
import os


def compute_histogram_and_cdf(image):
    """Compute histogram and CDF of an image."""
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    return hist, cdf, cdf_normalized


def create_mapping(cdf_input, cdf_reference):
    """Create a mapping from input image CDF to reference image CDF."""
    cdf_input_normalized = np.interp(cdf_input, (cdf_input.min(), cdf_input.max()), (0, 255))
    cdf_reference_normalized = np.interp(cdf_reference, (cdf_reference.min(), cdf_reference.max()), (0, 255))

    mapping = np.interp(cdf_input_normalized, cdf_reference_normalized, np.arange(256))
    return mapping.astype(np.uint8)


def apply_histogram_specification(image, mapping):
    """Apply histogram specification using the mapping."""
    return cv.LUT(image, mapping)


# Paths to input and reference images
input_image_path = "/home/student/220962019/opencv/lab2/image.jpg"
reference_image_path = "/home/student/220962019/opencv/lab2/reference_image.jpg"

# Debugging: Check if paths are correct
print("Input image path:", input_image_path)
print("Reference image path:", reference_image_path)

# Load the images
input_img = cv.imread(input_image_path)
reference_img = cv.imread(reference_image_path)

# Check if images were loaded correctly
if input_img is None:
    raise ValueError(f"Input image not loaded. Check the file path: {input_image_path}")

if reference_img is None:
    raise ValueError(f"Reference image not loaded. Check the file path: {reference_image_path}")

# Convert images to grayscale
input_img_gray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
reference_img_gray = cv.cvtColor(reference_img, cv.COLOR_BGR2GRAY)

# Compute histograms and CDFs
_, cdf_input, _ = compute_histogram_and_cdf(input_img_gray)
_, cdf_reference, _ = compute_histogram_and_cdf(reference_img_gray)

# Create the mapping from input to reference
mapping = create_mapping(cdf_input, cdf_reference)

# Apply histogram specification
input_img_specified = apply_histogram_specification(input_img_gray, mapping)

# Save the images
output_dir = "/home/student/220962019/opencv/lab2/"
cv.imwrite(os.path.join(output_dir, "input_image_gray.jpg"), input_img_gray)
cv.imwrite(os.path.join(output_dir, "reference_image_gray.jpg"), reference_img_gray)
cv.imwrite(os.path.join(output_dir, "input_image_specified.jpg"), input_img_specified)

print("Images saved successfully.")

#Q2 can be done as follows
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image, bins=256):
    """Calculate the histogram of an image."""
    hist, _ = np.histogram(image.flatten(), bins=bins, range=[0, 256])
    return hist

def calculate_cdf(hist):
    """Calculate the cumulative distribution function for a histogram."""
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # Normalize
    return cdf, cdf_normalized

def histogram_specification(input_image, reference_image):
    """Match the histogram of the input image to the reference image."""
    # Calculate the histogram and CDF of the input image
    input_hist = calculate_histogram(input_image)
    input_cdf, _ = calculate_cdf(input_hist)

    # Calculate the histogram and CDF of the reference image
    reference_hist = calculate_histogram(reference_image)
    reference_cdf, _ = calculate_cdf(reference_hist)

    # Create a lookup table to map input image pixel values to reference image pixel values
    lookup_table = np.zeros(256)
    g_j = 0
    for g_i in range(256):
        while g_j < 256 and input_cdf[g_i] > reference_cdf[g_j]:
            g_j += 1
        lookup_table[g_i] = g_j

    # Apply the mapping to the input image
    specified_image = cv2.LUT(input_image, lookup_table.astype(np.uint8))
    return specified_image

def show_images(input_image, reference_image, output_image):
    """Display input, reference, and output images."""
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('Input Image')
    plt.subplot(132), plt.imshow(reference_image, cmap='gray'), plt.title('Reference Image')
    plt.subplot(133), plt.imshow(output_image, cmap='gray'), plt.title('Specified Image')
    plt.show()

# Load input and reference images
input_image = cv2.imread('input_image.jpg', cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread('reference_image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform histogram specification
specified_image = histogram_specification(input_image, reference_image)

# Display the images
show_images(input_image, reference_image, specified_image)

