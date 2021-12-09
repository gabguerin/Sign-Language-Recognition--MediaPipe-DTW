import cv2
import mediapipe
import os
import pandas as pd

from models.sign_model import SignModel
from utils.mediapipe_utils import mediapipe_detection
from utils.landmark_utils import save_landmarks_from_video, load_array
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


if __name__ == "__main__":

    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
        os.path.dirname(file_name)
        for root, dirs, files in os.walk(os.path.join("data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    for video_name in videos_not_in_dataset:
        save_landmarks_from_video(video_name)

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    sign_dictionary = pd.DataFrame(columns=["name", "model", "distance"])
    for video_name in videos:
        sign_name = video_name.split("-")[0]
        path = os.path.join("data", "dataset", sign_name, video_name)

        pose_list = load_array(os.path.join(path, f"pose_{video_name}.pickle"))
        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))

        sign_dictionary = sign_dictionary.append(
            {
                "name": sign_name,
                "model": SignModel(pose_list, left_hand_list, right_hand_list),
                "distance": 0,
            },
            ignore_index=True,
        )

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(sign_dictionary)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, results, sign_detected, is_recording)

            # Record pressing s
            if cv2.waitKey(5) & 0xFF == ord("s"):
                sign_recorder.record()

            # Break pressing q
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
