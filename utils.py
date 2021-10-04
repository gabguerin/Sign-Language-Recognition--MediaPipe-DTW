import cv2
import mediapipe as mp
import pickle as pkl
import numpy as np

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_holistic = mp.solutions.holistic  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(235, 52, 86), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(52, 235, 103), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(235, 52, 86), thickness=1, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(52, 55, 235), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(235, 52, 86), thickness=1, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(52, 55, 201), thickness=2, circle_radius=2)
                             )


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()

def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
