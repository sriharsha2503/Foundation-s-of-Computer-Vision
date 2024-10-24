import cv2
import numpy as np

def detect_features(image):
    """
    Detect SIFT features in the given image.

    Parameters:
        image: The input image.

    Returns:
        keypoints: Detected keypoints.
        descriptors: Feature descriptors corresponding to the keypoints.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def lucas_kanade_tracking(prev_frame, next_frame, points):
    """
    Track feature points using the Lucas-Kanade optical flow method.

    Parameters:
        prev_frame: The previous frame (grayscale).
        next_frame: The next frame (grayscale).
        points: Points to track.

    Returns:
        good_old: Old points that were successfully tracked.
        good_new: New points after tracking.
    """
    # Calculate optical flow
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, points, None)

    # Select good points
    good_new = next_points[status == 1]
    good_old = points[status == 1]

    return good_old, good_new

def estimate_motion_model(old_points, new_points):
    """
    Estimate the motion model using the old and new points.

    Parameters:
        old_points: Old feature points.
        new_points: New feature points.

    Returns:
        matrix: The affine transformation matrix.
    """
    matrix, _ = cv2.estimateAffine2D(old_points, new_points)
    return matrix

def main(video_path):
    """
    Main function to run the Lucas-Kanade tracker on a video.

    Parameters:
        video_path: Path to the video file.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Read the first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to read the video.")
        return

    # Convert to grayscale
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect feature points in the first frame
    old_keypoints, _ = detect_features(old_gray)
    old_points = np.array([kp.pt for kp in old_keypoints], dtype=np.float32).reshape(-1, 1, 2)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

        # Track the feature points
        good_old, good_new = lucas_kanade_tracking(old_gray, new_gray, old_points)

        # Estimate the motion model
        motion_matrix = estimate_motion_model(good_old, good_new)

        # Draw the tracked points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            new_frame = cv2.circle(new_frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            new_frame = cv2.line(new_frame, (int(a), int(b)), (int(c), int(d)), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Tracking', new_frame)

        # Update the previous frame and points
        old_gray = new_gray.copy()
        old_points = good_new.reshape(-1, 1, 2)

        # Exit on ESC key
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your video file
    video_path = ""
    main(video_path)
