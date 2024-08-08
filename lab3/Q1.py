import cv2
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
