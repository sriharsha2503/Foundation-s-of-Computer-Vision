import cv2

cap = cv2.VideoCapture("/home/student/220962019/opencv/lab1/images/All 3 default dances.mp4")

out = cv2.VideoWriter()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) == ord('q'):
        break
