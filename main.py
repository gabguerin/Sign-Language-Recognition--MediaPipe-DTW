import cv2
import mediapipe
import utils
import os

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


if __name__ == "__main__":

    videos = [name.replace(".mp4", "") for name in os.listdir(os.path.join("data", "videos")) if name.endswith(".mp4")]
    dataset = [name for name in os.listdir(os.path.join("data", "dataset")) if not name.startswith(".")]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    for video in videos_not_in_dataset:
        save_landmarks_from_video(video + ".mp4")

    # Create a dictionary of reference signs
    sign_dictionary = {}
    for sign_name in dataset:
        path = os.path.join("data", "dataset", sign_name)

        pose_list = load_array(os.path.join(path, f"pose_{sign_name}.pickle"))
        left_hand_list = load_array(os.path.join(path, f"lh_{sign_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{sign_name}.pickle"))

        sign_dictionary[sign_name] = SignModel(pose_list, left_hand_list, right_hand_list)

    sign_distances = {k: 0 for k, _ in sign_dictionary}

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(sign_dictionary, sign_distances)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager(sign_distances)

    # Turn on the webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)

            # Process results
            sign_detected = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, results, sign_detected)

            # Record pressing s
            if cv2.waitKey(5) & 0xFF == ord('s'):
                sign_recorder.record()

            # Break pressing q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
