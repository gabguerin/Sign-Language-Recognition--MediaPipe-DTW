import cv2
import numpy as np

from dtw import dtw_distances
from utils.landmark_utils import extract_keypoints
from utils.mediapipe_utils import draw_landmarks

BLUE_COLOR = (245, 15, 15)
RED_COLOR = (15, 15, 245)


class WebcamManager(object):
    def __init__(self, sign_dictionary, seq_len=50):
        self.color = BLUE_COLOR
        self.recording = False
        self.seq_len = seq_len

        self.landmarks_list = []
        self.sign_dictionary = sign_dictionary

    def record(self):
        self.recording = True

    def update(self, frame, results):
        # Draw dataset
        draw_landmarks(frame, results)

        # Process results
        self.process_results(results)

        # Update the frame
        cv2.circle(frame, (30, 30), 20, self.color, -1)
        cv2.imshow('OpenCV Feed', frame)

    def process_results(self, results):
        if self.recording and len(self.sequence) < self.seq_len:
            self.record_movement(results)
        elif self.recording and len(self.sequence) == self.seq_len:
            self.compute_distances()

    def record_movement(self, results):
        # Record keypoints
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)

        # Red circle while recording
        self.color = RED_COLOR

    def compute_distances(self):
        sequence = np.array(self.landmarks_list)
        res = dtw_distances(sequence, signs)

        self.recording = False
        self.sequence = []
        self.color = BLUE_COLOR

