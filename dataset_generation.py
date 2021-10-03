from data_augmentation import DataAugmentationPipeline
from tensorflow.keras.utils import to_categorical
import os
from tqdm import tqdm
from utils import *


def generate_action(video):
    keypoint_list = []
    data_aug = DataAugmentationPipeline()

    cap = cv2.VideoCapture(os.path.join("Videos",video))
    # Set mediapipe model
    with mp.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Change distance to camera
                frame = data_aug.transform_image(frame)

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Rotate action
                data_aug.transform_results(results)

                # Store results
                keypoint_list.append(extract_keypoints(results))
            else:
                break
        cap.release()

    return keypoint_list


def generate_dataset(videos, nb_actions, seq_len):
    actions, labels = [], []
    for label in range(len(videos)):
        video = videos[label]
        if video not in os.listdir("landmarks"):
            os.makedirs("landmarks/"+video)

        for i in tqdm(range(nb_actions)):
            # Generate new action through data augmentation
            action = generate_action(video)

            # If the action is lower than seq_len we crop it
            # otherwise we copy last element of action until len(action) = seq_lenl
            if seq_len < len(action):
                action = action[:seq_len]
            else:
                action += action[-1] * (seq_len - len(action))

            # Saving landmarks
            path = os.path.join("landmarks", video, str(i + 1) + ".pickle")
            save_array([action, label], path)

            actions.append(action)
            labels.append(label)
    X = np.array(actions)
    y = to_categorical(labels).astype(int)
    return X, y

def load_dataset(videos):
    actions, labels = [], []
    for video in videos:
        path = os.path.join("landmarks", video)
        for f in os.listdir(path):
            action, label = load_array(os.path.join(path, f))
            actions.append(action)
            labels.append(label)
    X = np.array(actions)
    y = to_categorical(labels).astype(int)
    return X, y