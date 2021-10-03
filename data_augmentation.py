from scipy.linalg import expm, norm
import numpy as np
import cv2
import random


class DataAugmentationPipeline():
    def __init__(self):
        self.angle_x = random.choice([-1, 1]) * random.uniform(0, .04)
        self.angle_z = random.choice([-1, 1]) * random.uniform(0, .08)

        self.proportion = random.uniform(0.75, 1.25)

    def transform_results(self, results):
        axis = np.array((0, 1, (results.pose_landmarks.landmark[11].z + results.pose_landmarks.landmark[12].z) / 2))

        # Pose rotation
        self.rotate_landmarks(results.pose_landmarks.landmark, axis, self.angle_x)
        self.rotate_landmarks(results.pose_landmarks.landmark, axis, self.angle_z)

        # Left hand rotation & translation over left wrist
        if results.left_hand_landmarks:
            self.rotate_landmarks(results.left_hand_landmarks.landmark,
                                  axis,
                                  self.angle_x)
            self.rotate_landmarks(results.left_hand_landmarks.landmark,
                                  axis,
                                  self.angle_z)
            self.translate_landmarks(results.left_hand_landmarks.landmark,
                                     results.left_hand_landmarks.landmark[0],
                                     results.pose_landmarks.landmark[15])

        # Right hand rotation & translation over right wrist
        if results.right_hand_landmarks:
            self.rotate_landmarks(results.right_hand_landmarks.landmark,
                                  axis,
                                  self.angle_x)
            self.rotate_landmarks(results.right_hand_landmarks.landmark,
                                  axis,
                                  self.angle_z)
            self.translate_landmarks(results.right_hand_landmarks.landmark,
                                     results.right_hand_landmarks.landmark[0],
                                     results.pose_landmarks.landmark[16])

    def rotation_matrix(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    def rotate_landmarks(self, landmark_list, axis, angle):
        landmarks = np.array([np.array([res.x, res.y, res.z]) for res in landmark_list])

        M0 = self.rotation_matrix(axis, angle)
        rotated_landmarks = np.dot(M0, landmarks.T).T
        for i in range(len(landmark_list)):
            landmark_list[i].x = rotated_landmarks[i][0]
            landmark_list[i].y = rotated_landmarks[i][1]
            landmark_list[i].z = rotated_landmarks[i][2]

    def translate_landmarks(self, landmark_list, v0, v1):
        for i in range(len(landmark_list)):
            landmark_list[i].x += (v1.x - v0.x)
            landmark_list[i].y += (v1.y - v0.y)
            landmark_list[i].z += (v1.z - v0.z)

    def transform_image(self, image):
        """
            Function that rescale the frame to simulate distance to camera
        """
        height_0, width_0, _ = image.shape
        p = self.proportion

        scale = lambda x, p: int(x * p)
        height_1, width_1 = (scale(height_0, p), scale(width_0, p))

        img_resized = cv2.resize(image, (width_1, height_1), interpolation=cv2.INTER_AREA)

        lower_bounding = lambda x, p: int(x * (1 - p) / 2) + 1

        image = np.zeros((height_0, width_0, 3), np.uint8)

        if p < 1:
            # y0, y1, x0, x1 make the centered box where the downsized image is pasted
            y0, y1 = lower_bounding(height_0, p), lower_bounding(height_0, p) + height_1
            x0, x1 = lower_bounding(width_0, p), lower_bounding(width_0, p) + width_1
            image[y0:y1, x0:x1, :] = img_resized
        else:
            # y0, y1, x0, x1 make the centered box where the upsized image is cropped
            y0, y1 = lower_bounding(height_1, 1/p), lower_bounding(height_1, 1/p) + height_0
            x0, x1 = lower_bounding(width_1, 1/p), lower_bounding(width_1, 1/p) + width_0
            image = img_resized[y0:y1, x0:x1, :]
        return image