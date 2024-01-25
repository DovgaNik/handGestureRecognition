import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'data/gesture_recognizer.task'

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the video mode:
options = GestureRecognizerOptions(
    base_options = BaseOptions(model_path),
    running_mode = VisionRunningMode.VIDEO,
    num_hands = 2)
with GestureRecognizer.create_from_options(options) as recognizer:
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        gesture_recognition_result = recognizer.recognize_for_video(mp_image, int(capture.get(cv2.CAP_PROP_POS_MSEC)))
        
        try:
            top_gesture = gesture_recognition_result.gestures[0][0]
            print(top_gesture)
        except:
            pass




        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    
    capture.release()
    cv2.destroyAllWindows()