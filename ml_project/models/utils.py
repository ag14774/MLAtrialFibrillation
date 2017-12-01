import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from fastdtw import fastdtw
from numba import int32, jit, jitclass, prange
from pydtw import dtw1d

spec = [
    ('maxsize', int32),
    ('size', int32),
    ('arr', int32[:]),
    ('front', int32),
    ('rear', int32),
]


@jitclass(spec)
class Deque:
    def __init__(self, maxsize=100):
        self.maxsize = maxsize
        self.size = maxsize
        self.arr = np.empty((maxsize), dtype=np.int32)
        self.front = -1
        self.rear = 0

    def isEmpty(self):
        return self.front == -1

    def isFull(self):
        return ((self.front == 0 and self.rear == self.size - 1)
                or self.front == self.rear + 1)

    def append(self, key):
        if self.isFull():
            raise Exception("Full. Cannot insert element")
        if self.front == -1:
            self.front = 0
            self.rear = 0
        elif self.front == 0:
            self.front = self.size - 1
        else:
            self.front = self.front - 1
        self.arr[self.front] = key

    def pop(self):
        if self.isEmpty():
            raise Exception("Queue is empty. Cannot remove element")
        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        else:
            if self.front == self.size - 1:
                self.front = 0
            else:
                self.front = self.front + 1

    def popleft(self):
        if self.isEmpty():
            raise Exception("Queue is empty. Cannot remove element")
        if self.front == self.rear:
            self.front = -1
            self.rear = -1
        elif self.rear == 0:
            self.rear = self.size - 1
        else:
            self.rear = self.rear - 1

    def right(self):
        if self.isEmpty():
            raise Exception("Queue is empty. Cannot get element")
        return self.arr[self.front]

    def left(self):
        if self.isEmpty():
            raise Exception("Queue is empty. Cannot get element")
        return self.arr[self.rear]


@jit(nopython=True, nogil=True)
def sliding_window_minmax(arr, r):
    upper_bounds = np.empty_like(arr)
    lower_bounds = np.empty_like(arr)
    sorted_window_max = Deque(int(2 * r + 1))
    sorted_window_min = Deque(int(2 * r + 1))
    for i in range(2 * r + 1):
        # We have a more recent element that is larger than the
        # previous ones. So we can remove them.
        # If it's not larger, we still need to add it to the
        # queue because the older but larger elements will
        # eventually get out of the window
        while sorted_window_max.isEmpty(
        ) is False and arr[i] >= arr[sorted_window_max.right()]:
            sorted_window_max.pop()
        while sorted_window_min.isEmpty(
        ) is False and arr[i] <= arr[sorted_window_min.right()]:
            sorted_window_min.pop()
        sorted_window_max.append(i)
        sorted_window_min.append(i)
    n = len(arr)
    for i in range(r + 1, n - r):
        upper_bounds[i - 1] = arr[sorted_window_max.left()]
        lower_bounds[i - 1] = arr[sorted_window_min.left()]
        nextidx = i + r
        while sorted_window_max.isEmpty() is False and sorted_window_max.left(
        ) < i - r:
            sorted_window_max.popleft()
        while sorted_window_min.isEmpty() is False and sorted_window_min.left(
        ) < i - r:
            sorted_window_min.popleft()
        while sorted_window_max.isEmpty(
        ) is False and arr[nextidx] >= arr[sorted_window_max.right()]:
            sorted_window_max.pop()
        while sorted_window_min.isEmpty(
        ) is False and arr[nextidx] <= arr[sorted_window_min.right()]:
            sorted_window_min.pop()
        sorted_window_max.append(nextidx)
        sorted_window_min.append(nextidx)
    upper_bounds[n - r - 1] = arr[sorted_window_max.left()]
    lower_bounds[n - r - 1] = arr[sorted_window_min.left()]
    upper_bounds[0] = np.max(arr[0:r + 1])
    lower_bounds[0] = np.min(arr[0:r + 1])
    for i in range(1, r):
        upper_bounds[i] = upper_bounds[i - 1]
        lower_bounds[i] = lower_bounds[i - 1]
        if upper_bounds[i] < arr[i + r]:
            upper_bounds[i] = arr[i + r]
        if lower_bounds[i] > arr[i + r]:
            lower_bounds[i] = arr[i + r]
    upper_bounds[n - 1] = np.max(arr[n - 1 - r:n])
    lower_bounds[n - 1] = np.min(arr[n - 1 - r:n])
    for i in range(n - 2, n - r - 1, -1):
        upper_bounds[i] = upper_bounds[i + 1]
        lower_bounds[i] = lower_bounds[i + 1]
        if upper_bounds[i] < arr[i - r]:
            upper_bounds[i] = arr[i - r]
        if lower_bounds[i] > arr[i - r]:
            lower_bounds[i] = arr[i - r]
    return upper_bounds, lower_bounds


@jit(nopython=True, nogil=True)
def DTWDistanceWindow(s1, s2, w):
    DTW = np.full(
        (len(s1) + 1, len(s2) + 1), fill_value=np.inf, dtype=np.float32)
    DTW[0, 0] = 0
    # dists = cdist(s1, s2, 'cityblock')
    for i in range(0, len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w + 1)):
            DTW[i + 1, j + 1] = abs(s1[i] - s2[j]) + min(
                DTW[i, j + 1], DTW[i + 1, j], DTW[i, j])
    return DTW[-1, -1]


@jit(nopython=True, nogil=True)
def LB_Keogh(s1, s2, r):
    LB_sum = 0
    upper_bounds, lower_bounds = sliding_window_minmax(s2, r)
    for ind, i in enumerate(s1):
        lower_bound = lower_bounds[ind]
        upper_bound = upper_bounds[ind]
        if i > upper_bound:
            LB_sum = LB_sum + abs(i - upper_bound)
        elif i < lower_bound:
            LB_sum = LB_sum + abs(i - lower_bound)
    return LB_sum


@jit
def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag - 1]
    r = np.abs(r)
    return r, lag


@jit
def find_best_window(x, size=1000):
    best_i = 0
    best_r = 0
    best_l = 0
    for i in range(0, len(x) - size + 1):
        r, lag = autocorr(x[i:i+size])
        if r >= best_r:
            best_i = i
            best_r = r
            best_l = lag
    # print("Found best window with (r, l)=", best_r, best_l)
    # plt.plot(x[best_i:best_i+size])
    # plt.show()
    return best_i


@jit
def find_best_windows(X, size=1000):
    for i in range(len(X)):
        print("Selecting window for:", i)
        idx = find_best_window(X[i], size)
        X[i, 0:size] = X[i, idx:idx+size]
    return X[:, 0:size]


def scorer(estimator, X, y):
    ypred = estimator.predict(X)
    fscore = f1_score(y, ypred, average='micro')
    print("F1-Score: ", fscore)
    sys.stdout.flush()
    return fscore


@jit(nopython=True, nogil=True)
def DTWDistance(s1, s2, accuracy='exact', radius=1):
    dist = 0
    if accuracy == 'exact':
        cost_matrix = dtw1d(s1, s2)
        dist = cost_matrix[-1, -1]
    elif accuracy == 'window':
        dist = DTWDistanceWindow(s1, s2, radius)
    else:
        dist, _ = fastdtw(s1, s2, radius=radius)
    # print("Calculating DTW: ", dist)
    # sys.stdout.flush()
    return dist


@jit(nopython=True, nogil=True)
def fastdtw_wrapper(s1, s2, w):
    cost, _ = fastdtw(s1, s2, radius=w)
    return cost


@jit(nopython=True, nogil=True)
def dtw1d_wrapper(s1, s2, w):
    cost, _, _ = dtw1d(s1, s2)
    return cost[-1, -1]


@jit(nopython=True, nogil=True, parallel=True)
def calcLowerBounds(test, train, r):
    lbs = np.empty((test.shape[0], train.shape[0]), dtype=np.float32)
    for i in prange(test.shape[0]):
        print("Calculating lower bound for ", i)
        for j in range(train.shape[0]):
            lbs[i, j] = LB_Keogh(test[i], train[j], r)
    return lbs


@jit(nopython=True, nogil=True)
def knn(train, train_labels, test, w, accuracy=1):
    print("Calculating lower bounds...")
    # We can make the bound tighter by decreasing r
    lbs = calcLowerBounds(test, train, w // 2 - 1)
    preds = np.empty(test.shape[0])
    print("Starting 1-NN...")
    for i in range(test.shape[0]):
        min_dist = np.inf
        min_label = 0
        for j in range(train.shape[0]):
            if lbs[i, j] < min_dist:
                dist = DTWDistanceWindow(test[i], train[j], w)
                if dist < min_dist:
                    min_dist = dist
                    min_label = train_labels[j]
            else:
                print("Skipping DTW")
        preds[i] = min_label
    return preds


def fftanalysis(signal, rate=300):
    fftsignal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / rate)
    plt.plot(freqs, fftsignal)
    plt.show()
    idx = np.argmax(np.abs(fftsignal))
    print("Most important frequency: ", freqs[idx])
