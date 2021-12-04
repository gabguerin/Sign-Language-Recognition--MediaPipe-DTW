from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from models.sign_model import SignModel


def dtw_distances(recorded_sign: SignModel, sign_dictionary: dict):
    """
    Use DTW to compute similarity between the recorded sign & the reference signs

    :param recorded_sign: a SignModel object containing the data gathered during record
    :param sign_dictionary: A dictionary of SignModel objects containing the reference signs
    :return: Return a sign dictionary sorted by the distances from the recorded sign
    """
    # Embeddings of the recorded sign
    recorded_left_hand = recorded_sign.lh_embedding
    recorded_right_hand = recorded_sign.rh_embedding

    sign_distances = {}
    for sign_name, sign_model in sign_dictionary.items():
        # If the reference sign has the same number of hands compute fastdtw
        if (recorded_sign.has_left_hand == sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == sign_model.has_right_hand
        ):
            sign_left_hand = sign_model.lh_embedding
            sign_right_hand = sign_model.rh_embedding

            sign_distances[sign_name] = 0
            if recorded_sign.has_left_hand:
                sign_distances[sign_name] += fastdtw(recorded_left_hand, sign_left_hand)[0]
            if recorded_sign.has_right_hand:
                sign_distances[sign_name] += fastdtw(recorded_right_hand, sign_right_hand)[0]

        # If not, distance equals 10000
        else:
            sign_distances[sign_name] = 10000
    return dict(sorted(sign_distances.items(), key=lambda item: item[1]))
