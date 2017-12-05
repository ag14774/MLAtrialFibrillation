import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

from numba import jit


@jit(nopython=True)
def calc_refractory_period(sampling_rate=300):
    base_frequency = 250
    proportionality = sampling_rate / base_frequency
    return round(120 * proportionality)


@jit(nopython=True)
def effective_process_first_only(sample, process_first_only=-1):
    if process_first_only < 0:
        return sample.shape[0]
    return min(process_first_only, sample.shape[0])


@jit
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


@jit
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


@jit
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

        if qrs_peaks_indices.shape[0] > 0:
            last_qrs_index = qrs_peaks_indices[-1]
        else:
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


@jit
def isolate_qrs(s,
                num_of_qrs=5,
                sampling_rate=300,
                keep_full_refractory=False,
                scale_mode="after"):
    # if keep_full_refractory is True
    # we cut the full refractory_period
    # on each side of R.
    # if keep_full_refractory is False
    # we cut the half refractory_period
    # on each side
    if scale_mode == "before":
        s = scale(s, copy=False)
    refractory_period = calc_refractory_period(sampling_rate)
    qrs_peaks, _ = detect_qrs(s, sampling_rate)
    if len(qrs_peaks) > 1:
        # First peak is sometimes not very good
        qrs_peaks = qrs_peaks[1:]
    final_qrs_list = []
    if keep_full_refractory:
        width = refractory_period
    else:
        width = refractory_period // 2
    while len(final_qrs_list) < num_of_qrs:
        index_of_element_to_add = len(final_qrs_list) % len(qrs_peaks)
        final_qrs_list.append(qrs_peaks[index_of_element_to_add])
    new_s = np.empty((num_of_qrs, 2 * width))
    for i in range(len(final_qrs_list)):
        qrs_index = final_qrs_list[i]
        qrs_temp = s[max(0, qrs_index - width):qrs_index + width]
        # In case the signal is smaller. Pad with 0s. This ensures that
        # all QRS are always aligned
        if len(qrs_temp) < (2 * width):
            qrs_temp = np.pad(qrs_temp, (2 * width - len(qrs_temp)) // 2 + 1,
                              'median')
        new_s[i, :] = qrs_temp[:2 * width]
    if scale_mode == "after":
        new_s = scale(new_s, axis=1, copy=False)
    # Flatten the array
    return np.ravel(new_s)


def autocorr(x):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='same')
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag - 1]
    r = np.abs(r)
    return r, lag


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
