from fastdtw import fastdtw
import numpy as np


def dtw_distances(action, signs):
    arr = []
    action_lh, action_rh = action
    for sign in signs:
        # dist: sum of the dtw distance of left hand and right hand
        dist = 0
        # If the action and the sign use the same hands:
        #      compute the fasdtw distance
        # If not:
        #      distance is greater than 1000
        sign_lh, sign_rh = sign
        if len(sign_lh) > 0:
            if len(action_lh) > 0:
                dist += fastdtw(action_lh, sign_lh)[0]
            else:
                dist += 1000
        if len(sign_rh) > 0:
            if len(action_rh) > 0:
                dist += fastdtw(action_rh, sign_rh)[0]
            else:
                dist += 1000

        arr.append(dist)
    return arr
