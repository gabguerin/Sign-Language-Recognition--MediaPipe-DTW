import cv2
import numpy as np

from utils.mediapipe_utils import draw_landmarks

HEIGHT = 1100
VIDEO_WIDTH = 1100
DICT_WIDTH = 300


class WebcamManager(object):
    def __init__(self, sign_distances):
        self.sign_distances = sign_distances

        self.video_panel = np.ones((HEIGHT, VIDEO_WIDTH))
        self.dict_panel = np.array((HEIGHT, DICT_WIDTH))

    def update(self, frame, results):
        # Draw landmarks
        frame = draw_landmarks(frame, results)

        # Update panels
        self.video_panel = cv2.resize(frame, (VIDEO_WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        self.update_dict_panel()

        # Concatenate the two panels
        image = np.concatenate((self.video_panel, self.dict_panel), axis=1)

        # Update the frame
        cv2.circle(image, (30, 30), 20, self.color, -1)
        cv2.imshow('OpenCV Feed', frame)

    def update_dict_panel(self):
        for i in range(len(self.sign_distances)):
            pos = (20, 40 * (i + 1))
            self.dict_panel = cv2.putText(self.dict_panel, f"{i + 1}. {self.dictionary[i]} ",
                                          pos,
                                          cv2.FONT_HERSHEY_TRIPLEX,
                                          1,
                                          (255, 255, 255),
                                          1, cv2.LINE_AA)

