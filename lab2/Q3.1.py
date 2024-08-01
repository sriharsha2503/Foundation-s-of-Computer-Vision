import cv2 as cv
import numpy as np

def resize_image(image, width=None, height=None, fx=None, fy=None, interpolation=cv.INTER_LINEAR):
    """Resize the image to a specific width and height or scale factor."""
    if width is not None and height is not None:
        # Resize to a specific width and height
        resized = cv.resize(image, (width, height), interpolation=interpolation)
    elif fx is not None and fy is not None:
        # Resize by scale factors
        resized = cv.resize(image, None, fx=fx, fy=fy, interpolation=interpolation)
    else:
        raise ValueError("Either width and height or scale factors fx and fy must be provided.")
    return resized

def crop_image(image, x, y, width, height):
    """Crop the image to a specific region defined by (x, y, width, height)."""
    return image[y:y+height, x:x+width]

# Load the image
image_path = "/home/student/220962019/opencv/lab2/image.jpg"
image = cv.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Resize the image
resized_image = resize_image(image, width=400, height=300)  # Resize to 400x300 pixels

# Crop the image
# Define the ROI (x, y, width, height)
roi_x, roi_y, roi_width, roi_height = 50, 50, 200, 200
cropped_image = crop_image(image, roi_x, roi_y, roi_width, roi_height)

# Save the images
cv.imwrite("/home/student/220962019/opencv/lab2/resized_image.jpg", resized_image)
cv.imwrite("/home/student/220962019/opencv/lab2/cropped_image.jpg", cropped_image)

print("Images saved successfully.")
