import cv2
import numpy as np

# Load the video stream
cap = cv2.VideoCapture('')

# Define parameters
corner_threshold = 0.01  # Threshold for corner detection
window_size = 5  # Window size for computing optical flow
re_detection_threshold = 50  # Threshold to re-detect corners if points are lost

# Harris Corner Detector Implementation
def harris_corner_detector(img, window_size=3, k=0.04):
    # Calculate gradients
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Apply Gaussian filter
    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 0)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 0)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 0)

    # Calculate Harris response
    det = (Sxx * Syy) - (Sxy * Sxy)
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    # Threshold to find corners
    corners = np.zeros_like(R)
    corners[R > corner_threshold * R.max()] = 1
    return np.argwhere(corners == 1)

# Lucas-Kanade Optical Flow Calculation (without built-in functions)
def lucas_kanade_optical_flow(old_img, new_img, points, window_size=5):
    flow_vectors = []

    # Calculate gradients
    Ix = cv2.Sobel(old_img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(old_img, cv2.CV_64F, 0, 1, ksize=3)
    It = new_img - old_img

    half_window = window_size // 2

    for point in points:
        y, x = point.ravel()
        if x - half_window < 0 or y - half_window < 0 or x + half_window >= old_img.shape[1] or y + half_window >= old_img.shape[0]:
            continue  # Skip points near borders

        # Define window
        Ix_window = Ix[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
        Iy_window = Iy[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()
        It_window = It[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1].flatten()

        # Form the A matrix and b vector
        A = np.vstack((Ix_window, Iy_window)).T
        b = -It_window

        # Solve for velocity vector [vx, vy] using least-squares
        nu, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        flow_vectors.append((x + nu[0], y + nu[1]))

    return np.array(flow_vectors)

# Read the first frame and initialize feature points
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
points = harris_corner_detector(old_gray)

# Main loop for tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    new_points = lucas_kanade_optical_flow(old_gray, new_gray, points, window_size=window_size)

    # Draw the tracked points
    for new, old in zip(new_points, points):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        cv2.circle(frame, (int(a), int(b)), 3, (0, 255, 0), -1)

    # Update old frame and points
    old_gray = new_gray.copy()
    points = harris_corner_detector(new_gray) if len(new_points) < re_detection_threshold else new_points

    # Display the frame
    cv2.imshow('KLT Tracker (Custom)', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
