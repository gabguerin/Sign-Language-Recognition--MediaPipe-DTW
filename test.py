import cv2
import mediapipe

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.test_dataset_utils import load_test_dataset, load_test_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from test_sign_recorder import TestSignRecorder
from webcam_manager import WebcamManager


if __name__ == "__main__":
    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    videos = load_dataset()
    test_videos = load_test_dataset()
    reference_signs = load_reference_signs(videos)
    print()
    test_signs = load_test_reference_signs(test_videos)


    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)
    test_sign_recorder = SignRecorder(test_signs)
    
    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # # Set up the Mediapipe environment
    # with mediapipe.solutions.holistic.Holistic(
    #     min_detection_confidence=0.5, min_tracking_confidence=0.5
    # ) as holistic:
    #     while cap.isOpened():

    #         # Read frame from the video file
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         # Make detections
    #         image, results = mediapipe_detection(frame, holistic)
    #         # print(results.face_landmarks)
    #         # print(results.pose_landmarks)
    #         # print(len(results.pose_landmarks[0]))
    #         # Process results
    #         sign_detected, is_recording = sign_recorder.process_results(results)

    #         # Update the frame (draw landmarks & display result)
    #         webcam_manager.update(frame, results, sign_detected, is_recording)

    #         pressedKey = cv2.waitKey(1) & 0xFF
            