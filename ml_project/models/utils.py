import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import butter, lfilter
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale

from biosppy import utils
from biosppy.signals import ecg, tools
from numba import jit


def check_X_tuple(X):
    try:
        a, b = X
    except Exception as e:
        a = X
        b = np.array([])
    return a, b


@jit
def lastNonZero(s):
    for i in range(len(s) - 1, -1, -1):
        if s[i] != 0:
            return i
    return 0


def biosspyX(X, sampling_rate=300, show=False, verbose=True):
    data = []
    for i, x in enumerate(X):
        data.append(ecg.ecg(x, sampling_rate, show))
        if verbose:
            if i % 300 == 0:
                print("Analysing sample ", i, "...Please wait...")
                sys.stdout.flush()
    return data


def signal_stats(signal):
    try:
        return tools.signal_stats(signal)
    except Exception:
        args = (0, 0, 0, 0, 0, 0, 0, 0)
        names = ('mean', 'median', 'max', 'var', 'std_dev', 'abs_dev',
                 'kurtosis', 'skewness')

        return utils.ReturnTuple(args, names)


def extract_data(biooutput, sampling_rate=300, min_length=200):
    filtered_signal = biooutput["filtered"]
    median_template = np.median(biooutput["templates"], axis=0)
    mean_template = np.mean(biooutput["templates"], axis=0)

    median_template_stats = signal_stats(median_template)
    mean_template_stats = signal_stats(mean_template)
    heartrate_stats = signal_stats(biooutput["heart_rate"])

    peak_values = filtered_signal[biooutput["rpeaks"]]
    peak_stats = signal_stats(peak_values)

    new_median_template = np.empty((min_length))
    new_median_template[:len(median_template)] = median_template
    new_mean_template = np.empty((min_length))
    new_mean_template[:len(mean_template)] = mean_template
    if len(median_template) < min_length:
        new_median_template = np.pad(
            median_template,
            round((min_length - len(median_template)) / 2),
            mode='median')[:min_length]
        new_mean_template = np.pad(
            mean_template,
            round((min_length - len(mean_template)) / 2),
            mode='mean')[:min_length]

    return (filtered_signal, new_median_template, new_mean_template,
            median_template_stats, mean_template_stats, heartrate_stats,
            peak_stats)


def template_min_length(data):
    maxlength = 0
    for x in data:
        length = x["templates"].shape[1]
        if length > maxlength:
            maxlength = length
    return maxlength


def featurevector(processed_signal, sampling_rate=300, min_length=200):
    results = extract_data(
        processed_signal, sampling_rate=sampling_rate, min_length=200)
    filtered_signal = results[0]
    median_template = results[1]
    mean_template = results[2]
    median_template_stats = results[3]
    mean_template_stats = results[4]
    heartrate_stats = results[5]
    peak_stats = results[6]

    features = np.empty((len(median_template) + len(mean_template) + 4*8))
    offset = len(median_template)
    features[:offset] = median_template
    features[offset:offset+offset] = mean_template

    features[offset+offset + 0] = median_template_stats["mean"]
    features[offset+offset + 1] = median_template_stats["median"]
    features[offset+offset + 2] = median_template_stats["max"]
    features[offset+offset + 3] = median_template_stats["var"]
    features[offset+offset + 4] = median_template_stats["std_dev"]
    features[offset+offset + 5] = median_template_stats["abs_dev"]
    features[offset+offset + 6] = median_template_stats["kurtosis"]
    features[offset+offset + 7] = median_template_stats["skewness"]

    features[offset+offset + 8] = mean_template_stats["mean"]
    features[offset+offset + 9] = mean_template_stats["median"]
    features[offset+offset + 10] = mean_template_stats["max"]
    features[offset+offset + 11] = mean_template_stats["var"]
    features[offset+offset + 12] = mean_template_stats["std_dev"]
    features[offset+offset + 13] = mean_template_stats["abs_dev"]
    features[offset+offset + 14] = mean_template_stats["kurtosis"]
    features[offset+offset + 15] = mean_template_stats["skewness"]

    features[offset+offset + 16] = heartrate_stats["mean"]
    features[offset+offset + 17] = heartrate_stats["median"]
    features[offset+offset + 18] = heartrate_stats["max"]
    features[offset+offset + 19] = heartrate_stats["var"]
    features[offset+offset + 20] = heartrate_stats["std_dev"]
    features[offset+offset + 21] = heartrate_stats["abs_dev"]
    features[offset+offset + 22] = heartrate_stats["kurtosis"]
    features[offset+offset + 23] = heartrate_stats["skewness"]

    features[offset+offset + 24] = peak_stats["mean"]
    features[offset+offset + 25] = peak_stats["median"]
    features[offset+offset + 26] = peak_stats["max"]
    features[offset+offset + 27] = peak_stats["var"]
    features[offset+offset + 28] = peak_stats["std_dev"]
    features[offset+offset + 29] = peak_stats["abs_dev"]
    features[offset+offset + 30] = peak_stats["kurtosis"]
    features[offset+offset + 31] = peak_stats["skewness"]

    return filtered_signal, features


@jit
def fixBaseline(s, sampling_rate=300):
    small_step = round(0.2 * sampling_rate)
    big_step = round(0.6 * sampling_rate)
    if small_step % 2 == 0:
        small_step = small_step + 1
    if big_step % 2 == 0:
        big_step = big_step + 1
    filter1 = scipy.signal.medfilt(s, kernel_size=small_step)
    filter2 = scipy.signal.medfilt(filter1, kernel_size=big_step)
    return s - filter2


@jit(nopython=True)
def calc_refractory_period(sampling_rate=300):
    base_frequency = 250
    proportionality = sampling_rate / base_frequency
    return round(50 * proportionality)


@jit(nopython=True)
def RRIntervals(qrs_indices, sampling_rate=300):
    RR = np.diff(qrs_indices) / sampling_rate
    return RR


# def compute_dRR(RRs, sampling_rate=300):
#     RRs2 = np.hstack((RRs[1:].reshape(-1, 1), RRs[:len(RRs) - 1].reshape(
#         -1, 1)))
#     dRRs = np.zeros((len(RRs) - 1, 1))
#     for i in range(len(RRs2)):
#         if np.sum(RRs2[i, :] < 0.5) >= 1:
#             dRRs[i, 0] = 2 * (RRs2[i, 0] - RRs2[i, 1])
#         elif np.sum(RRs2[i, :] > 1) >= 1:
#             dRRs[i, 0] = 0.5 * (RRs2[i, 0] - RRs2[i, 1])
#         else:
#             dRRs[i, 0] = (RRs2[i, 0] - RRs2[i, 1])
#     return dRRs

# def BPcount(sZ):
#     bdc = 0
#     BC = 0
#     pdc = 0
#     PC = 0
#     for i in range(-2, 3):
#         bdc = np.sum(np.diag(sZ, i) != 0, axis=0)
#         pdc = np.sum(np.diag(sZ, i), axis=0)
#         BC = BC + bdc
#         PC = PC + pdc
#         sZ = sZ - np.diag(np.diag(sZ, i), i)
#     return BC, PC, sZ

# def metrics(dRR):
#     dRR = np.hstack((dRR[1:len(dRR)], dRR[0:len(dRR) - 1]))
#     OCmask = 0.02
#     os = np.sum(np.abs(dRR) <= OCmask, axis=1)
#     OriginCount = np.sum(os == 2, axis=0)
#     OLmask = 1.5
#     dRRnew = np.array([]).reshape(-1, dRR.shape[1])
#     for i in range(dRR.shape[0]):
#         if np.sum(np.abs(dRR[i, :]) >= OLmask, axis=0) == 0:
#             dRRnew = np.vstack((dRRnew, dRR[i, :]))
#     if len(dRRnew) == 0:
#         dRRnew = np.array([0, 0]).reshape(-1, dRR.shape[1])
#     print(dRRnew)
#     bin_c = np.arange(-0.58, 0.621, 0.04)
#     Z, _ = np.histogramdd(dRRnew, [bin_c, bin_c])
#     print(Z)
#
#     Z[13, 14:16] = 0
#     Z[14:16, 13:17] = 0
#     Z[16, 14:16] = 0
#
#     Z2 = Z[15:30, 15:30]
#     BC12, PC12, sZ2 = BPcount(Z2)
#     Z[15:30, 15:30] = sZ2
#
#     Z3 = Z[15:30, 0:15]
#     Z3 = np.fliplr(Z3)
#     BC11, PC11, sZ3 = BPcount(Z3)
#     Z[15:30, 0:15] = np.fliplr(sZ3)
#
#     Z4 = Z[0:15, 0:15]
#     BC10, PC10, sZ4 = BPcount(Z4)
#     Z[0:15, 0:15] = sZ4
#
#     Z1 = Z[0:15, 15:30]
#     Z1 = np.fliplr(Z1)
#     BC9, PC9, sZ1 = BPcount(Z1)
#     Z[0:15, 15:30] = np.fliplr(sZ1)
#
#     BC5 = np.sum(np.sum(Z[0:15, 13:17] != 0, axis=0), axis=0)
#     PC5 = np.sum(np.sum(Z[0:15, 13:17], axis=0), axis=0)
#
#     BC7 = np.sum(np.sum(Z[15:30, 13:17] != 0, axis=0), axis=0)
#     PC7 = np.sum(np.sum(Z[15:30, 13:17], axis=0), axis=0)
#
#     BC6 = np.sum(np.sum(Z[13:17, 0:15] != 0, axis=0), axis=0)
#     PC6 = np.sum(np.sum(Z[13:17, 0:15], axis=0), axis=0)
#
#     BC8 = np.sum(np.sum(Z[13:17, 15:30] != 0, axis=0), axis=0)
#     PC8 = np.sum(np.sum(Z[13:17, 15:30], axis=0), axis=0)
#
#     Z[13:17, :] = 0
#     Z[:, 13:17] = 0
#
#     BC2 = np.sum(np.sum(Z[0:13, 0:13] != 0, axis=0), axis=0)
#     PC2 = np.sum(np.sum(Z[0:13, 0:13], axis=0), axis=0)
#
#     BC1 = np.sum(np.sum(Z[0:13, 17:30] != 0, axis=0), axis=0)
#     PC1 = np.sum(np.sum(Z[0:13, 17:30], axis=0), axis=0)
#
#     BC3 = np.sum(np.sum(Z[17:30, 0:13] != 0, axis=0), axis=0)
#     PC3 = np.sum(np.sum(Z[17:30, 0:13], axis=0), axis=0)
#
#     BC4 = np.sum(np.sum(Z[17:30, 17:30] != 0, axis=0), axis=0)
#     PC4 = np.sum(np.sum(Z[17:30, 17:30], axis=0), axis=0)
#
#     IrrEv = (BC1 + BC2 + BC3 + BC4 + BC5 + BC6 + BC7 + BC8 + BC9 + BC10 +
#              BC11 + BC12)
#     PACEv = (PC1 - BC1) + (PC2 - BC2) + (PC3 - BC3) + (PC4 - BC4) + (
#         PC5 - BC5) + (PC6 - BC6) + (PC10 - BC10) - (PC7 - BC7) - (
#             PC8 - BC8) - (PC12 - BC12)
#
#     return OriginCount, IrrEv, PACEv


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


# @jit
def detect_qrs(ecg_data_raw, signal_frequency=300, skip_bandpass=False):
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
    refractory_period = round(50 * proportionality)  # 120 default
    possible_t_period = round(90 * proportionality)
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
    qrs_peaks_values = np.array([], dtype=float)
    noise_peaks_indices = np.array([], dtype=int)

    # Measurements filtering - 0-15 Hz band pass filter.
    if skip_bandpass is True:
        filtered_ecg_measurements = ecg_data_raw.copy()
    else:
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
                qrs_peaks_values = np.append(qrs_peaks_values,
                                             detected_peaks_value)

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

    final_mask = np.ones(len(qrs_peaks_indices), dtype=bool)
    if (qrs_peaks_indices.shape[0] > 1):
        last_qrs_index = qrs_peaks_indices[0]
        last_qrs_value = qrs_peaks_values[0]
        for i in range(1, len(qrs_peaks_indices)):
            current_qrs_index = qrs_peaks_indices[i]
            current_qrs_value = qrs_peaks_values[i]
            # print(current_qrs_value)
            if (current_qrs_index - last_qrs_index <= possible_t_period):
                if (current_qrs_value > last_qrs_value):
                    final_mask[i - 1] = False
                    last_qrs_index = current_qrs_index
                    last_qrs_value = current_qrs_value
                else:
                    final_mask[i] = False
            else:
                last_qrs_index = current_qrs_index
                last_qrs_value = current_qrs_value

    # print(qrs_peaks_indices)
    # print(qrs_peaks_indices[final_mask])
    return qrs_peaks_indices[final_mask], noise_peaks_indices


def isolate_qrs(s,
                num_of_qrs=5,
                sampling_rate=300,
                refractory_fraction=1.0,
                skip_bandpass=False,
                scale_mode="after"):
    # if keep_full_refractory is True
    # we cut the full refractory_period
    # on each side of R.
    # if keep_full_refractory is False
    # we cut the half refractory_period
    # on each side
    if scale_mode == "before":
        try:
            s = scale(s, copy=False)
        except Exception:
            print("Scaling causes division by 0...Skipping")
            sys.stdout.flush()
    refractory_period = calc_refractory_period(sampling_rate)
    qrs_peaks, _ = detect_qrs(s, sampling_rate, skip_bandpass=skip_bandpass)
    if len(qrs_peaks) > 1:
        # First peak is sometimes not very good
        qrs_peaks = qrs_peaks[1:]
    width = round(refractory_period * refractory_fraction)
    # Add cleanup code to select spikes with
    # low variance only
    final_qrs_list = []
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
        try:
            new_s = scale(new_s, axis=1, copy=False)
        except Exception:
            print("Scaling causes division by 0...Skipping")
            sys.stdout.flush()
    # Flatten the array
    return np.ravel(new_s)


def autocorr(x, mode='same'):
    n = x.size
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode=mode)
    acorr = result[n // 2 + 1:] / (x.var() * np.arange(n - 1, n // 2, -1))
    lag = np.abs(acorr).argmax() + 1
    r = acorr[lag - 1]
    r = np.abs(r)
    # plt.plot(acorr)
    # plt.show()
    return r, lag, acorr


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
