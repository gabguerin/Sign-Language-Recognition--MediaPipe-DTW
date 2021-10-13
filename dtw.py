from fastdtw import fastdtw
import numpy as np


def dtw_distances(action, signs):
    arr = []
    for sign in signs:
        arr.append(fastdtw(action, sign)[0])
    return arr
