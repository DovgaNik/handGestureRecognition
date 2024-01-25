import cv2
import mediapipe as mp

model_path = 'data/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options = BaseOptions(model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 10)


with GestureRecognizer.create_from_options(options) as recognizer:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(capture.get(cv2.CAP_PROP_POS_MSEC)))
        
        for hand_landmark in gesture_recognition_result.hand_landmarks:
            try:
                for landmark in hand_landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                print("Good frame")

                connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                            (0, 5), (5, 9), (9, 13), (13, 17), (0, 17), 
                            (5, 6), (6, 7), (7, 8),
                            (9, 10), (10, 11), (11, 12),
                            (13, 14), (14, 15), (15, 16), 
                            (17, 18), (18, 19), (19, 20)]
                for connection in connections:
                    start_point = (int(hand_landmark[connection[0]].x * w), int(hand_landmark[connection[0]].y * h))
                    end_point = (int(hand_landmark[connection[1]].x * w), int(hand_landmark[connection[1]].y * h))
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            except Exception as e:
                print(f"Bad frame {e}")

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    capture.release()
    cv2.destroyAllWindows()
