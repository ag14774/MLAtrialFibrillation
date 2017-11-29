import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from fastdtw import fastdtw
from pydtw import dtw1d


def scorer(estimator, X, y):
    ypred = estimator.predict(X)
    fscore = f1_score(y, ypred, average='micro')
    print("F1-Score: ", fscore)
    sys.stdout.flush()
    return fscore


def DTWDistance(s1, s2, accuracy='exact'):
    dist = 0
    if accuracy == 'exact':
        cost_matrix = dtw1d(s1, s2)
        dist = cost_matrix(len(s1)-1, len(s2)-1)
    else:
        dist, _ = fastdtw(s1, s2)
    # print("Calculating DTW: ", dist)
    # sys.stdout.flush()
    return dist


def fftanalysis(signal, rate=300):
    fftsignal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1/rate)
    plt.plot(freqs, fftsignal)
    plt.show()
    idx = np.argmax(np.abs(fftsignal))
    print("Most important frequency: ", freqs[idx])
