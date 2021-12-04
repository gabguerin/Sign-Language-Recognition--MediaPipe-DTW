import cv2
import numpy as np

from utils.mediapipe_utils import draw_landmarks


BLUE_COLOR = (245, 242, 176)
RED_COLOR = (25, 35, 240)

HEIGHT = 1200


class WebcamManager(object):
    """
    Object that displays the Webcam output, draws the landmarks detected and
    outputs the sign prediction
    """
    def __init__(self):
        self.sign_detected = ""

    def update(self, frame: np.ndarray, results, sign_detected: str, is_recording: bool):
        self.sign_detected = sign_detected

        # Draw landmarks
        frame = draw_landmarks(frame, results)

        WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
        # Resize frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Write result if there is
        frame = self.draw_text(frame)

        # Chose circle color
        color = BLUE_COLOR
        if is_recording:
            color = RED_COLOR

        # Update the frame
        cv2.circle(frame, (30, 30), 20, color, -1)
        cv2.imshow('OpenCV Feed', frame)

    def draw_text(self,
                  frame,
                  font=cv2.FONT_HERSHEY_COMPLEX,
                  font_size=2,
                  font_thickness=4,
                  offset=20,
                  bg_color=(245, 242, 176, 0.85)):
        window_w = int(HEIGHT * len(frame[0]) / len(frame))

        (text_w, text_h), _ = cv2.getTextSize(self.sign_detected, font, font_size, font_thickness)

        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset

        cv2.rectangle(frame, (0, text_y - offset), (window_w, HEIGHT), bg_color, -1)
        cv2.putText(
            frame,
            self.sign_detected,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (20, 20, 20),
            font_thickness,
        )
        return frame
