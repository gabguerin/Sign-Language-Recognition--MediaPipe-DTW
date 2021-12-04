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
    # Sequence of landmarks
    landmarks = os.listdir("landmarks")[1:]


    left_panel, right_panel = st.columns(2)

    with left_panel:
        show_mediapipe = st.checkbox('Afficher Mediapipe')
        video_panel = st.image([])
        record_sign = st.checkbox('Enregistrer')

    with right_panel:
        st.subheader("Dictionnaire")

        sign_names = list(map(lambda x: x.replace(".pickle", ""), landmarks))
        st.text("- " + sign_names[0])
        st.text("- " + sign_names[1])
        st.text("- " + sign_names[2])
        st.text("- " + sign_names[3])
        st.text("- " + sign_names[4])

        st.subheader("Résultat")
        result_txt = st.code("")

        st.subheader("Détails")
        result_df_txt = st.dataframe()


    signs = []
    for landmark in landmarks:
        path = os.path.join("landmarks", landmark)
        signs.append(utils.load_array(path))

    lh_list, rh_list = [], []
    BLUE_COLOR = (255, 255, 255)
    RED_COLOR = (25, 25, 255)
    color = BLUE_COLOR


    results_df = pd.DataFrame({"signs": [], "distances": []})

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.25) as holistic:
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
                color = RED_COLOR

            else:
                lh_list, rh_list = utils.load_array("save.pickle")
                if len(lh_list) > 0 or len(rh_list) > 0:
                    action = np.array([lh_list, rh_list])
                    distances = dtw_distances(action, signs)

                    results_df = pd.DataFrame({"signs": sign_names, "distances": distances}).sort_values(by=["distances"])
                    print(results_df)
                    signs_sorted = results_df.signs.values
                    result_txt.code(signs_sorted[0].replace(".pickle", ""))
                    result_df_txt.dataframe(results_df)

                    utils.save_array([[], []], "save.pickle")
                    lh_list, rh_list = [], []

                color = BLUE_COLOR

            # REC circle
            cv2.circle(frame, (30, 30), 20, color, -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_panel.image(frame)

