import cv2
import mediapipe

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


if __name__ == "__main__":
    with open('log.txt', 'w') as file:
    # Write the new content to the end of the file
        file.write("New log\n")
    # Create dataset of the videos where landmarks have not been extracted yet
    videos = load_dataset() # 랜드마크 추출하는거임. 문제 없음.

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    reference_signs = load_reference_signs(videos)

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(0)
    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, results, sign_detected, is_recording)
            if sign_detected:
                last_line = None

                # Open the file for reading
                with open('log.txt', 'r') as file:
                    # Read all lines into a list
                    lines = file.readlines()
                    # Get the last line
                    if lines:
                        last_line = lines[-1].strip()  # Strip to remove newline characters

                # Compare the last message with sign_detected
                if last_line != str(sign_detected):
                    # Open the file for appending and write the new message
                    with open('log.txt', 'a') as file:
                        
                        file.write(str(sign_detected))
                        file.write("\n")
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break
            

        cap.release()
        cv2.destroyAllWindows()
