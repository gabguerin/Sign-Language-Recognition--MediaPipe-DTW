import cv2
import mediapipe
from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager

if __name__ == "__main__":
    # Create dataset of the videos where landmarks have not been extracted yet
    videos = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    reference_signs = load_reference_signs(videos)

    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)
    webcam_manager = WebcamManager()
    # Open the video file
    cap = cv2.VideoCapture('data/videos/Bonjour/Bonjour-irWEaNF6kmo.mp4')

    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            # Read frame from the video
            ret, frame = cap.read()
            # print("Frame shape:", frame.shape)
            if not ret:
                break  # Break the loop if we've reached the end of the video

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)
    
    
            webcam_manager.update(frame, results, sign_detected, is_recording)

            # if sign_detected:
            #     text = "Sign Detected: " + sign_detected  # Modify this line with the sign_detected result
            #     cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Draw landmarks and display result (you can modify this part for custom drawing)
            # For example, to draw landmarks:
            # mediapipe.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mediapipe.solutions.pose.POSE_CONNECTIONS)
            # mediapipe.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mediapipe.solutions.holistic.HAND_CONNECTIONS)
            # mediapipe.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mediapipe.solutions.holistic.HAND_CONNECTIONS)
            # Display the processed frame
            # cv2.imshow('Processed Frame', image)

            # Press 'q' to exit the loop
            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        if sign_detected:
            print(sign_detected)
        else: 
            print("No sign")