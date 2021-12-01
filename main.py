import cv2
import mediapipe
import utils
import os

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array
from webcam_manager import WebcamManager

if __name__ == "__main__":

    videos = [name.replace(".mp4", "") for name in os.listdir("data/videos") if name.endswith(".mp4")]
    dataset = [name for name in os.listdir("data/dataset") if not name.startswith(".")]

    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    for video in videos_not_in_dataset:
        save_landmarks_from_video(video + ".mp4")

    # Create a dictionary of reference signs
    sign_dictionary = {}
    for sign_name in dataset:
        path = os.path.join("data/dataset", sign_name)

        left_hand_landmarks = load_array(os.path.join(path, f"lh_{sign_name}.pickle"))
        right_hand_landmarks = load_array(os.path.join(path, f"rh_{sign_name}.pickle"))

        sign_dictionary[sign_name] = SignModel(left_hand_landmarks, right_hand_landmarks)

    webcam_manager = WebcamManager(sign_dictionary)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)

            # Update frame and process results
            webcam_manager.update(frame, results)

            # Break pressing q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Record pressing s
            if cv2.waitKey(5) & 0xFF == ord('s'):
                webcam_manager.record()

        cap.release()
        cv2.destroyAllWindows()
