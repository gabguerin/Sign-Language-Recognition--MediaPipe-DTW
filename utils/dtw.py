import pandas as pd
from fastdtw import fastdtw
import numpy as np
from models.sign_model import SignModel
from tslearn.metrics import soft_dtw


def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding
    rec_pose = recorded_sign.pose_embedding  # Add this line

    for idx, row in reference_signs.iterrows():
        # Initialize the row variables
        ref_sign_name, ref_sign_model, _ = row

        # If the reference sign has the same number of hands compute fastdtw
        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding
            ref_pose = ref_sign_model.pose_embedding  # Add this line

            # if recorded_sign.has_left_hand:
            #     row["distance"] += list(fastdtw(rec_left_hand, ref_left_hand))[0]
            # if recorded_sign.has_right_hand:
            #     row["distance"] += list(fastdtw(rec_right_hand, ref_right_hand))[0]
            # row["distance"] += (fastdtw(rec_pose, ref_pose))
            distance, _ = fastdtw(rec_pose, ref_pose)
            row["distance"] += distance
        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])


def soft_dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame):
    """
    Use SoftDTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param reference_signs: pd.DataFrame
                            columns : name, dtype: str
                                      sign_model, dtype: SignModel
                                      distance, dtype: float64
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding
    rec_pose = recorded_sign.pose_embedding  # Add this line
    # if True:
    #   raise ValueError(rec_pose)
    for idx, row in reference_signs.iterrows():
        # Initialize the row variables
        ref_sign_name, ref_sign_model, _ = row

        # If the reference sign has the same number of hands compute softdtw
        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            ref_left_hand = ref_sign_model.lh_embedding
            ref_right_hand = ref_sign_model.rh_embedding
            ref_pose = ref_sign_model.pose_embedding  # Add this line

            # if recorded_sign.has_left_hand:
            #     row["distance"] += soft_dtw(rec_left_hand, ref_left_hand)
            # if recorded_sign.has_right_hand:
            #     row["distance"] += soft_dtw(rec_right_hand, ref_right_hand)
            
            # Use rec_pose and ref_pose in your calculations as needed
            # For example:
            # if True:
            #     raise ValueError(rec_pose, ref_pose)
            row["distance"] += soft_dtw(rec_pose, ref_pose)

        # If not, distance equals infinity
        else:
            row["distance"] = np.inf
    return reference_signs.sort_values(by=["distance"])