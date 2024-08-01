import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv.imread("/home/student/220962019/opencv/lab2/image.jpg")

# Check if the image was loaded correctly
if img is None:
    raise ValueError("Image not loaded. Check the file path.")

# Convert to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Perform histogram equalization
img_eq = cv.equalizeHist(img_gray)

# Calculate histograms before and after equalization
hist_before, bins_before = np.histogram(img_gray.flatten(), 256, [0, 256])
hist_after, bins_after = np.histogram(img_eq.flatten(), 256, [0, 256])

# Calculate CDFs before and after equalization
cdf_before = hist_before.cumsum()
cdf_after = hist_after.cumsum()
cdf_normalized_before = cdf_before * float(hist_before.max()) / cdf_before.max()
cdf_normalized_after = cdf_after * float(hist_after.max()) / cdf_after.max()

# Plot histograms and CDFs
plt.figure(figsize=(14, 8))

# Original Image and Histogram
plt.subplot(2, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(cdf_normalized_before, color='b')
plt.hist(img_gray.flatten(), 256, [0, 256], color='r', alpha=0.7)
plt.xlim([0, 256])
plt.title('Before Equalization')
plt.legend(('CDF', 'Histogram'), loc='upper left')

# Equalized Image and Histogram
plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(cdf_normalized_after, color='b')
plt.hist(img_eq.flatten(), 256, [0, 256], color='r', alpha=0.7)
plt.xlim([0, 256])
plt.title('After Equalization')
plt.legend(('CDF', 'Histogram'), loc='upper left')

plt.tight_layout()
plt.show()
