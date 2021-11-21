import cv2
import mediapipe
import utils
import numpy as np
import os
import streamlit as st
from extract_landmarks import extract_frame_landmarks
from dtw import dtw_distances
import pandas as pd


if __name__ == '__main__':

    left_panel, right_panel = st.columns(2)

    with left_panel:
        show_mediapipe = st.checkbox('Montrer Mediapipe')
        video_panel = st.image([])
        record_sign = st.checkbox('Enregistrer')

    with right_panel:
        st.subheader("Dictionnaire")

        videos = os.listdir("Videos")[2:]
        first_sign_txt = st.text("- " + videos[0].replace(".mp4", ""))
        second_sign_txt = st.text("- " + videos[1].replace(".mp4", ""))
        third_sign_txt = st.text("- " + videos[2].replace(".mp4", ""))
        fourth_sign_txt = st.text("- " + videos[3].replace(".mp4", ""))
        fifth_sign_txt = st.text("- " + videos[4].replace(".mp4", ""))

    landmarks = os.listdir("landmarks")[1:]
    signs = []
    for landmark in landmarks:
        path = os.path.join("landmarks",landmark)
        signs.append(utils.load_array(path))

    # Sequence of landmarks
    #utils.save_array([[], []], "save.pickle")
    lh_list, rh_list = [], []
    seq_len = 50
    count = 0

    blue_color = (255, 25, 16)
    red_color = (24, 44, 255)
    color = blue_color

    results_df = pd.DataFrame({"signs": [], "distances": []})

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.25, min_tracking_confidence=0.25) as holistic:
        while True:
            _, frame = cap.read()
            frame, results = utils.mediapipe_detection(frame, holistic)

            if show_mediapipe:
                utils.draw_landmarks(frame, results)

            if record_sign:
                lh, rh = extract_frame_landmarks(results)
                lh_list.append(lh)
                rh_list.append(rh)

                utils.save_array([lh_list, rh_list], "save.pickle")
                color = red_color

            else:
                lh_list, rh_list = utils.load_array("save.pickle")
                if len(lh_list) > 0 or len(rh_list) > 0:
                    action = np.array([lh_list, rh_list])
                    distances = dtw_distances(action, signs)

                    results_df = pd.DataFrame({"signs": landmarks, "distances": distances}).sort_values(by=["distances"])
                    print(results_df)
                    signs_sorted = results_df.signs.values
                    first_sign_txt.text("1. " + signs_sorted[0].replace(".pickle", ""))
                    second_sign_txt.text("2. " + signs_sorted[1].replace(".pickle", ""))
                    third_sign_txt.text("3. " + signs_sorted[2].replace(".pickle", ""))
                    fourth_sign_txt.text("4. " + signs_sorted[3].replace(".pickle", ""))
                    fifth_sign_txt.text("5. " + signs_sorted[4].replace(".pickle", ""))

                    utils.save_array([[], []], "save.pickle")
                    lh_list, rh_list = [], []

                color = blue_color

            # REC circle
            cv2.circle(frame, (30, 30), 20, color, -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_panel.image(frame)

