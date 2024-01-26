import cv2
import mediapipe as mp
import numpy as np

gesture_mapping = {
    "None": "None",
    "Closed_Fist": "Closed fist",
    "Open_Palm": "Open palm",
    "Pointing_Up": "Pointing up",
    "Thumb_Down": "Thumb down",
    "Thumb_Up": "Thumb up",
    "Victory": "Victory",
    "ILoveYou": "I love you"
}

model_path = 'data/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=3,
    min_hand_presence_confidence=0.4)

with GestureRecognizer.create_from_options(options) as recognizer:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(capture.get(cv2.CAP_PROP_POS_MSEC)))

        for i in range(len(gesture_recognition_result.hand_landmarks)):

            hand_landmark = gesture_recognition_result.hand_landmarks[i]
            try:
                for landmark in hand_landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                connections = [(0, 1), (1, 2), (2, 3), (3, 4),
                               (0, 5), (5, 9), (9, 13), (13, 17), (0, 17), (0, 9), (0, 13),
                               (5, 6), (6, 7), (7, 8),
                               (9, 10), (10, 11), (11, 12),
                               (13, 14), (14, 15), (15, 16),
                               (17, 18), (18, 19), (19, 20)]

                for connection in connections:
                    start_point = (int(hand_landmark[connection[0]].x * w), int(hand_landmark[connection[0]].y * h))
                    end_point = (int(hand_landmark[connection[1]].x * w), int(hand_landmark[connection[1]].y * h))
                    cv2.line(frame, start_point, end_point, (0, 0, 255), 1)

                text_position = (int(hand_landmark[0].x * w), int(hand_landmark[0].y * h))
                text_to_display = f"{gesture_mapping[gesture_recognition_result.gestures[i][0].category_name]} {int(np.around(gesture_recognition_result.gestures[i][0].score, 2) * 100)}%"
                cv2.putText(frame, text_to_display, text_position, cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
            except Exception as e:
                print(f"Bad frame: {e}")

        cv2.imshow('Hand gesture recognizer project', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()
