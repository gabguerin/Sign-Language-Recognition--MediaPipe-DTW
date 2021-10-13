import cv2
import mediapipe
import utils
import numpy as np
import os
from extract_landmarks import extract_landmarks, extract_keypoints
from dtw import dtw_distances
import pandas as pd


if __name__ == '__main__':

    videos = os.listdir("Videos")[1:]
    landmarks = os.listdir("landmarks")[1:]
    for video in videos:
        if video[:-4]+'.pickle' not in landmarks:
            extract_landmarks(video)

    signs = []
    landmarks = os.listdir("landmarks")[1:]
    for landmark in landmarks:
        path = os.path.join("landmarks",landmark)
        signs.append(utils.load_array(path))


    action = utils.load_array("oui_val.pickle")

    distances = dtw_distances(action, signs)

    df = pd.DataFrame({"signs":landmarks, "distances":distances}).sort_values(by=["distances"])
    df.to_csv("distances_from_oui.csv")


"""

    # Sequence of landmarks
    sequence = []
    seq_len = 40
    recording = False

    blue_color = (245, 25, 16)
    red_color = (24, 44, 245)
    color = blue_color

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)

            # Draw landmarks
            utils.draw_landmarks(image, results)

            if recording and len(sequence) < seq_len:
                # Record keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)

                # Red circle while recording
                color = red_color

            elif recording and len(sequence) == seq_len:
                sequence = np.array(sequence)
                res = dtw_distances(sequence, signs)

                print(res)

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
        
"""