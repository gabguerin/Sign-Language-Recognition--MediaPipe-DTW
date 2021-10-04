from fastdtw import fastdtw
import numpy as np


def dtw_distances(action, signs):
    arr = []
    n = len(action)
    for sign in signs:
        m = len(sign)
        if n < m:
            arr.append(dtw_between_actions(action, sign[:n], n))
        else:
            arr.append(dtw_between_actions(action[:m], sign, m))
    return arr

def dtw_between_actions(action1, action2, seq_len):
    dist_ = []
    nb_keypoints = len(action1[0])
    for i in range(nb_keypoints):
        dist_.append(fastdtw(action1[i][:], action2[i][:], radius=seq_len)[0])
    return np.mean(dist_)