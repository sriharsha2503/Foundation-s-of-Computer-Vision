import cv2
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
