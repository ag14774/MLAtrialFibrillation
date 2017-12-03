import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement


def bandpass_filter(data,
                    lowcut=0.0,
                    highcut=15.0,
                    signal_freq=300,
                    filter_order=1):
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


def extract_qrs_random(s, random_state, sampling_rate=300):
    base_frequency = 250
    proportionality = sampling_rate / base_frequency
    refractory_period = round(120 * proportionality)
    length_of_qrs = 2 * refractory_period
    peaks, _ = detect_qrs(s, sampling_rate)
    s = bandpass_filter(s, 0.0, 15.0, sampling_rate, 1)
    # print(peaks)
    random_state = check_random_state(random_state)
    qrss = sample_without_replacement(len(peaks), 1, random_state=random_state)
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


def transform_data(X, y, QRSList=None, random_state=None, sampling_rate=300):
    base_frequency = 250
    proportionality = sampling_rate / base_frequency
    refractory_period = round(120 * proportionality)
    length_of_qrs = 2 * refractory_period
    if QRSList is None:
        Xnew = np.empty((X.shape[0], length_of_qrs), dtype=np.float32)
    else:
        Xnew = np.empty(
            (X.shape[0], length_of_qrs * len(QRSList)), dtype=np.float32)
    for i, sample in enumerate(X):
        # print(i)
        if QRSList is None:
            temp = extract_qrs_random(
                sample, random_state, sampling_rate=sampling_rate)
        else:
            temp = extract_qrs(sample, QRSList, sampling_rate=sampling_rate)
        Xnew[i, :] = np.ravel(temp)
        # plt.plot(Xnew[i, :])
        # plt.show()
    return Xnew, y


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


def fftanalysis(signal, rate=300):
    fftsignal = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(signal), d=1 / rate)
    plt.bar(freqs, fftsignal)
    plt.show()
    idx = np.argmax(np.abs(fftsignal))
    print("Most important frequency: ", freqs[idx])
