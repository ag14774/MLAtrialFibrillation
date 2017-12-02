import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

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


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    """
    Method responsible for creating and applying Butterworth filter.
    :param deque data: raw data
    :param float lowcut: filter lowcut frequency value
    :param float highcut: filter highcut frequency value
    :param int signal_freq: signal frequency in samples per second (Hz)
    :param int filter_order: filter order
    :return array: filtered data
    """
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y


def findpeaks(data, spacing=1, limit=None):
    """
    Janko Slavic peak detection algorithm and implementation.
    https://github.com/jankoslavic/py-tools/tree/master/findpeaks
    Finds peaks in `data` which are of `spacing` width and >=`limit`.
    :param ndarray data: data
    :param float spacing: minimum spacing to the next peak (should be >=1)
    :param float limit: peaks should have value greater or equal
    :return array: detected peaks indexes array
    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start:start + len]  # before
        start = spacing
        h_c = x[start:start + len]  # central
        start = spacing + s + 1
        h_a = x[start:start + len]  # after
        peak_candidate = np.logical_and(peak_candidate,
                                        np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def detect_qrs(ecg_data_raw, signal_frequency=300):
    """
    Python Offline ECG QRS Detector based on the Pan-Tomkins algorithm.

    Michał Sznajder (Jagiellonian University) - technical contact
    (msznajder@gmail.com) Marta Łukowska (Jagiellonian University)
    The module is offline Python implementation of QRS complex detection in
    the ECG signal based on the Pan-Tomkins algorithm: Pan J, Tompkins W.J.,
    A real-time QRS detection algorithm, IEEE Transactions on Biomedical
    Engineering, Vol. BME-32, No. 3, March 1985, pp. 230-236.
    The QRS complex corresponds to the depolarization of the right and left
    ventricles of the human heart. It is the most visually obvious part of the
    ECG signal. QRS complex detection is essential for time-domain ECG signal
    analyses, namely heart rate variability. It makes it possible to compute
    inter-beat interval (RR interval) values that correspond to the time
    between two consecutive R peaks. Thus, a QRS complex detector is an
    ECG-based heart contraction detector. Offline version detects QRS complexes
    in a pre-recorded ECG signal dataset (e.g. stored in .csv format).
    This implementation of a QRS Complex Detector is by no means a certified
    medical tool and should not be used in health monitoring. It was created
    and used for experimental purposes in psychophysiology and psychology.
    You can find more information in module documentation:
    https://github.com/c-labpl/qrs_detector
    If you use these modules in a research project, please consider citing it:
    https://zenodo.org/record/583770
    If you use these modules in any other project, please refer to MIT
    open-source license.
    MIT License
    Copyright (c) 2017 Michał Sznajder, Marta Łukowska
    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    """

    base_frequency = 250
    proportionality = signal_frequency / base_frequency
    findpeaks_spacing = round(50 * proportionality)
    refractory_period = round(120 * proportionality)
    integration_window = round(15 * proportionality)

    filter_lowcut = 0.0
    filter_highcut = 15.0
    filter_order = 1
    findpeaks_limit = 0.35
    qrs_peak_filtering_factor = 0.125
    noise_peak_filtering_factor = 0.125
    qrs_noise_diff_weight = 0.25

    qrs_peak_value = 0.0
    noise_peak_value = 0.0
    threshold_value = 0.0

    # Detection results.
    qrs_peaks_indices = np.array([], dtype=int)
    noise_peaks_indices = np.array([], dtype=int)

    # Measurements filtering - 0-15 Hz band pass filter.
    filtered_ecg_measurements = bandpass_filter(
        ecg_data_raw,
        lowcut=filter_lowcut,
        highcut=filter_highcut,
        signal_freq=signal_frequency,
        filter_order=filter_order)
    filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

    # Derivative - provides QRS slope information.
    differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)

    # Squaring - intensifies values received in derivative.
    squared_ecg_measurements = differentiated_ecg_measurements**2

    # Moving-window integration.
    integrated_ecg_measurements = np.convolve(squared_ecg_measurements,
                                              np.ones(integration_window))

    # Fiducial mark - peak detection on integrated measurements.
    detected_peaks_indices = findpeaks(
        data=integrated_ecg_measurements,
        limit=findpeaks_limit,
        spacing=findpeaks_spacing)

    detected_peaks_values = integrated_ecg_measurements[detected_peaks_indices]

    for detected_peak_index, detected_peaks_value in zip(
            detected_peaks_indices, detected_peaks_values):

        try:
            last_qrs_index = qrs_peaks_indices[-1]
        except IndexError:
            last_qrs_index = 0

        # After a valid QRS complex detection, there is a 200 ms refractory
        # period before next one can be detected.
        diff = detected_peak_index - last_qrs_index
        refp = refractory_period
        if diff > refp or qrs_peaks_indices.size == 0:

            # Peak must be classified either as a noise peak or a QRS peak.
            # To be classified as a QRS peak it must exceed dynamically
            # set threshold value.
            if detected_peaks_value > threshold_value:
                qrs_peaks_indices = np.append(qrs_peaks_indices,
                                              detected_peak_index)

                # Adjust QRS peak value used later for setting
                # QRS-noise threshold.
                qrs_peak_value = (
                    qrs_peak_filtering_factor * detected_peaks_value) + (
                        1 - qrs_peak_filtering_factor) * qrs_peak_value
            else:
                noise_peaks_indices = np.append(noise_peaks_indices,
                                                detected_peak_index)

                # Adjust noise peak value used later for setting QRS-noise
                # threshold.
                noise_peak_value = (
                    noise_peak_filtering_factor * detected_peaks_value) + (
                        1 - noise_peak_filtering_factor) * noise_peak_value

            # Adjust QRS-noise threshold value based on previously detected
            # QRS or noise peaks value.
            threshold_value = (noise_peak_value) + qrs_noise_diff_weight * (
                qrs_peak_value - noise_peak_value)

    return qrs_peaks_indices, noise_peaks_indices


def extract_qrs(s, qrss, sampling_rate=300):
    base_frequency = 250
    proportionality = sampling_rate / base_frequency
    refractory_period = round(120 * proportionality)
    length_of_qrs = 2 * refractory_period
    peaks, _ = detect_qrs(s, sampling_rate)
    s = bandpass_filter(s, 0.0, 15.0, sampling_rate, 1)
    # print(peaks)
    if (qrss[-1] >= len(peaks)):
        print("Failsafe enabled...")
        sys.stdout.flush()
        qrss = qrss.copy()
        qrss.fill(0)
    result = np.empty((len(qrss), length_of_qrs), dtype=np.float32)
    for i in range(len(qrss)):
        qrorder = qrss[i]
        qrindex = peaks[qrorder]
        temp = s[max(0, qrindex - refractory_period):
                 qrindex + refractory_period]
        temp = np.pad(temp, (length_of_qrs - len(temp)) // 2 + 1, 'median')
        result[i, :] = temp[:length_of_qrs]
    result = scale(result, axis=1, copy=False)
    return result


def transform_data(X, y, QRSList, sampling_rate=300):
    base_frequency = 250
    proportionality = sampling_rate / base_frequency
    refractory_period = round(120 * proportionality)
    length_of_qrs = 2 * refractory_period
    Xnew = np.empty(
        (X.shape[0], length_of_qrs * len(QRSList)), dtype=np.float32)
    for i, sample in enumerate(X):
        # print(i)
        Xnew[i, :] = np.ravel(
            extract_qrs(sample, QRSList, sampling_rate=sampling_rate))
        # plt.plot(Xnew[i, :])
        # plt.show()
    return Xnew, y


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


def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag - 1]
    r = np.abs(r)
    return r, lag


def find_best_window(x, size=1000):
    best_i = 0
    best_r = 0
    # best_l = 0
    for i in range(0, len(x) - size + 1):
        r, lag = autocorr(x[i:i + size])
        if r >= best_r:
            best_i = i
            best_r = r
            # best_l = lag
    # print("Found best window with (r, l)=", best_r, best_l)
    # plt.plot(x[best_i:best_i+size])
    # plt.show()
    return best_i


def find_best_windows(X, size=1000):
    for i in range(X.shape[0]):
        print("Selecting window for:", i)
        idx = find_best_window(X[i], size)
        X[i, 0:size] = X[i, idx:idx + size]
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


@jit
def fastdtw_wrapper(s1, s2, radius=1):
    cost, _ = fastdtw(s1, s2, radius=radius)
    return cost


@jit
def fastdtwQRS(s1, s2, radius=1, length_of_qrs=50):
    s1 = s1.reshape(-1, length_of_qrs)
    s2 = s2.reshape(-1, length_of_qrs)
    metric = partial(fastdtw_wrapper, radius=radius)
    dists = cdist(s1, s2, metric=metric)
    # print(dists.shape)
    return np.min(dists)


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
