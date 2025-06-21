import cv2

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Testing camera index {i}")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Camera Index {i}', frame)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        cap.release()
