from model import Seq2seqClassifier
from dataset_generation import generate_dataset, load_dataset
from sklearn.model_selection import train_test_split
import cv2
import mediapipe
import utils
import numpy as np


if __name__ == '__main__':
    dload = False # Either load or generate dataset
    mload = False # Either load or generate model

    videos = ["Bienvenue - LSF.mp4",
              "Non - LSF.mp4",
              "Oui - LSF.mp4"]

    nb_actions = 30
    seq_len = 40
    # ( Left hand keypoints + Right hand kp + Pose kp ) * 3 coordinates x, y, z
    nb_keypoints = (21 + 21 + 9) * 3

    X, y = load_dataset(videos) if dload \
        else generate_dataset(videos, nb_actions, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

    # Input shape = (seq_len, nb_keypoints * nb_actions
    model = Seq2seqClassifier(input_shape=(seq_len, nb_keypoints), num_classes=len(videos))
    if mload:
        model.load('')
    else:
        model.train(X_train, y_train, epochs=1000)

    # Sequence of landmarks
    sequence = []
    recording = False

    blue_color = (245, 25, 16)
    red_color = (24, 44, 245)
    color = blue_color

    cap = cv2.VideoCapture(0)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)

            # Draw landmarks
            utils.draw_landmarks(image, results)

            if recording and len(sequence) < seq_len:
                # Record keypoints
                keypoints = utils.extract_keypoints(results)
                sequence.append(keypoints)

                # Red circle while recording
                color = red_color

            elif recording and len(sequence) == seq_len:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(videos[np.argmax(res)], ';', res)

                recording = False
                sequence = []
                color = blue_color

            # REC circle
            cv2.circle(image, (30, 30), 20, color, -1)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break pressing q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Record pressing s
            if cv2.waitKey(5) & 0xFF == ord('s'):
                recording = True

        cap.release()
        cv2.destroyAllWindows()