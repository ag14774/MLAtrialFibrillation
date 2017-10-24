import numpy as np

def probs2labels(probs):
    return np.argmax(probs, axis=1)