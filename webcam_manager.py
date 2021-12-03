import cv2

from utils.mediapipe_utils import draw_landmarks


HEIGHT = 1200
WIDTH = 1400


class WebcamManager(object):
    """
    Object that displays the Webcam output, draws the landmarks detected
    and outputs the sign prediction
    """
    def __init__(self, sign_detected: str):
        self.sign_detected = sign_detected

    def update(self, frame, results, sign_detected):
        self.sign_detected = sign_detected

        # Draw landmarks
        frame = draw_landmarks(frame, results)

        # Resize frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Write result if there is
        frame = self.draw_text(frame)

        # Update the frame
        cv2.circle(frame, (30, 30), 20, self.color, -1)
        cv2.imshow('OpenCV Feed', frame)

    def draw_text(self,
                  frame,
                  font=cv2.FONT_HERSHEY_DUPLEX,
                  font_size=3,
                  font_thickness=2,
                  offset=20,
                  bg_color=(232, 254, 255, 0.8)):
        (text_w, text_h), _ = cv2.getTextSize(self.sign_detected, font, font_size, font_thickness)

        text_x, text_y = int((WIDTH + text_w) / 2), HEIGHT - text_h - offset

        cv2.rectangle(frame, (0, text_y - offset), (WIDTH, HEIGHT), bg_color, -1)
        cv2.putText(
            frame,
            self.sign_detected,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
        )
        return frame
