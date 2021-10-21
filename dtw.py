from fastdtw import fastdtw
import numpy as np


def clean(action):
    action_lh, action_rh = action
    clean_action_lh, clean_action_rh = [], []
    for frame in action_lh:
        if np.sum(frame) > 0:
            clean_action_lh.append(frame)
    for frame in action_rh:
        if np.sum(frame) > 0:
            clean_action_rh.append(frame)
    return clean_action_lh, clean_action_rh

def dtw_distances(action, signs):
    arr = []
    action_lh, action_rh = clean(action)
    #print(action_rh)
    for sign in signs:
        # dist: sum of the dtw distance of left hand and right hand
        dist = 0
        # If the action and the sign use the same hands:
        #      compute the fasdtw distance
        # If not:
        #      distance is greater than 1000
        sign_lh, sign_rh = clean(sign)
        #print(sign_rh)
        if np.sum(sign_lh) > 0:
            if np.sum(action_lh) > 0:
                dist += fastdtw(action_lh, sign_lh)[0]
            else:
                dist += 10000
        else:
            if np.sum(action_lh) > 0:
                dist += 10000
        if np.sum(sign_rh) > 0:
            if np.sum(action_rh) > 0:
                dist += fastdtw(action_rh, sign_rh)[0]
            else:
                dist += 10000
        else:
            if np.sum(action_rh) > 0:
                dist += 10000

        arr.append(dist)
    return arr
