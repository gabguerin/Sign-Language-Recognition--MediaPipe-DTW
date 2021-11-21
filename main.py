import cv2
import mediapipe
import utils
import numpy as np
import os
from extract_landmarks import extract_video_landmarks, extract_frame_landmarks
from dtw import dtw_distances
import pandas as pd


if __name__ == '__main__':

    videos = os.listdir("Videos")[2:]
    landmarks = os.listdir("landmarks")
    for video in videos:
        if video[:-4]+'.pickle' not in landmarks\
                and video[-4:] in ['.mp4', '.mov']:
            extract_video_landmarks(video)

    signs = []

    """landmarks = ["Bonjour - LSF.pickle",
                 "Ca va - LSF.pickle",
                 "S il vous plait - LSF.pickle",
                 "Oui - LSF.pickle"]"""
    for landmark in landmarks[1:]:
        path = os.path.join("landmarks",landmark)
        signs.append(utils.load_array(path))

    """
    landmarks = ["Bonjour_val.pickle",
                 "Ca_va_val.pickle",
                 "silvousplait_val.pickle"]
    for landmark in landmarks:
        path = os.path.join("landmarks",landmark)
        action = utils.load_array(path)

        distances = dtw_distances(action, signs)

        df = pd.DataFrame({"signs":landmarks, "distances":distances}).sort_values(by=["distances"])

        print(landmark)
        print(df)

    """

    # Sequence of landmarks
    lh_list, rh_list = [], []
    seq_len = 50
    count = 0
    recording = False

    blue_color = (245, 25, 16)
    red_color = (24, 44, 245)
    color = blue_color

    results_df = pd.DataFrame({"signs": [], "distances": []})

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            video_panel, results = utils.mediapipe_detection(frame, holistic)

            # Draw landmarks
            utils.draw_landmarks(video_panel, results)

            if recording and count < seq_len:
                # Store results
                lh, rh = extract_frame_landmarks(results)
                lh_list.append(lh)
                rh_list.append(rh)

                count +=1

                # Red circle while recording
                color = red_color

            elif recording and count == seq_len:
                action = np.array([lh_list, rh_list])
                distances = dtw_distances(action, signs)

                results_df = pd.DataFrame({"signs": landmarks[1:], "distances": distances}).sort_values(by=["distances"])
                print(results_df)

                count = 0
                recording = False
                lh_list, rh_list = [], []
                color = blue_color

            # REC circle
            cv2.circle(video_panel, (30, 30), 20, color, -1)

            """
            # Black panel where we output the results
            results_panel = np.zeros((video_panel.shape[0], 200, 3)).astype(np.uint8)

            for idx, row in results_df.iterrows():
                pos = (20, 40 * (idx + 1))
                if pos[1] > video_panel.shape[0]:
                    break

                results_panel = cv2.putText(results_panel, f"{idx + 1}. {row['signs']} ({row['distances']})",
                                            pos,
                                            cv2.FONT_HERSHEY_TRIPLEX,
                                            1,
                                            (255, 255, 255),
                                            1, cv2.LINE_AA)

            frame = np.concatenate((video_panel, results_panel), axis=1)
            """
            # Show to screen
            cv2.imshow('OpenCV Feed', video_panel)

            # Break pressing q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Record pressing s
            if cv2.waitKey(5) & 0xFF == ord('s'):
                recording = True

        cap.release()
        cv2.destroyAllWindows()

        #"""