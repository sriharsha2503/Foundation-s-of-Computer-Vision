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

