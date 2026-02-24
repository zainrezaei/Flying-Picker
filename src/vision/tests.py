from picamera2 import Picamera2
import cv2

def video():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()

    while True:
        frame = picam2.capture_array()

        cv2.imshow("webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    video()