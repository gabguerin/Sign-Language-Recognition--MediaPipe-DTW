import cv2
import mediapipe as mp


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Draw left hand connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.left_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )
    # Draw right hand connections
    image = mp_drawing.draw_landmarks(
        image,
        landmark_list=results.right_hand_landmarks,
        connections=mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing.DrawingSpec(
            color=(232, 254, 255), thickness=1, circle_radius=4
        ),
        connection_drawing_spec=mp_drawing.DrawingSpec(
            color=(255, 249, 161), thickness=2, circle_radius=2
        ),
    )
    return image
