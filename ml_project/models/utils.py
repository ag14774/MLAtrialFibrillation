import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelmax, butter, lfilter
from scipy.stats import stats
from sklearn.metrics import f1_score

from biosppy import plotting, utils
from biosppy.signals import tools as st
from biosppy.signals.ecg import (correct_rpeaks, extract_heartbeats,
                                 hamilton_segmenter)
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
        temp = ecg2(
            x,
            sampling_rate=sampling_rate,
            show=show,
            peakdetector=qrs_detector)
        args = (X[i], temp["ts"], temp["filtered"], temp["rpeaks"],
                temp["templates_ts"], temp["templates"], temp["heart_rate_ts"],
                temp["heart_rate"])
        names = ('original', 'ts', 'filtered', 'rpeaks', 'templates_ts',
                 'templates', 'heart_rate_ts', 'heart_rate')
        temp = utils.ReturnTuple(args, names)
        data.append(temp)
        if verbose:
            if i % 300 == 0:
                print("Analysing sample ", i, "...Please wait...")
                sys.stdout.flush()
    return data


def signal_stats(signal):
    try:
        # mean, median, max, min, std, skewness, kurtosis
        # check inputs
        if signal is None:
            raise TypeError("Please specify an input signal.")

        # ensure numpy
        signal = np.array(signal)

        # mean
        mean = np.mean(signal)

        # median
        median = np.median(signal)

        # maximum amplitude abs
        maxAmpAbs = np.abs(signal - mean).max()

        # minimum amplitude abs
        minAmpAbs = np.abs(signal - mean).min()

        # maximum amplitude
        maxAmp = (signal - mean).max()

        # minimum amplitude
        minAmp = (signal - mean).min()

        # variance
        sigma2 = signal.var(ddof=1)

        # standard deviation
        sigma = signal.std(ddof=1)

        # absolute deviation
        ad = np.sum(np.abs(signal - median))

        # kurtosis
        kurt = stats.kurtosis(signal, bias=False)

        # skweness
        skew = stats.skew(signal, bias=False)

        # output
        args = (mean, median, maxAmpAbs, minAmpAbs, maxAmp, minAmp, sigma2,
                sigma, ad, kurt, skew)
        names = ('mean', 'median', 'maxabs', 'minabs', 'max', 'min', 'var',
                 'std_dev', 'abs_dev', 'kurtosis', 'skewness')

        return utils.ReturnTuple(args, names)
    except Exception:
        args = (0, 0, 0, 0, 0, 0, 0, 0)
        names = ('mean', 'median', 'max', 'var', 'std_dev', 'abs_dev',
                 'kurtosis', 'skewness')

        return utils.ReturnTuple(args, names)


def qrs_detector(signal, sampling_rate=300):
    grad_sqr = (np.gradient(signal))**2
    grad_sqr = gaussian_filter(grad_sqr, sigma=40)
    grad_integrated = np.convolve(grad_sqr,
                                  np.ones(round(sampling_rate / 250 * 10)))
    filtered_grad = gaussian_filter(grad_integrated, sigma=30)
    # plt.plot(filtered_grad)
    # plt.plot(signal)
    # plt.show()
    candidates = argrelmax(filtered_grad)[0]
    maxima = filtered_grad[candidates]
    mask = (np.abs(maxima) > 0.25 * np.median(np.abs(maxima)))
    return utils.ReturnTuple((candidates[mask], ), ('rpeaks', ))


def ecg2(signal=None,
         sampling_rate=1000.,
         show=True,
         peakdetector=hamilton_segmenter,
         peak_params={}):
    """Process a raw ECG signal and extract relevant signal features using
    default parameters.
    Parameters
    ----------
    signal : array
        Raw ECG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    show : bool, optional
        If True, show a summary plot.
    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    order = int(0.3 * sampling_rate)
    filtered, _, _ = st.filter_signal(
        signal=signal,
        ftype='FIR',
        band='bandpass',
        order=order,
        frequency=[3, 45],
        sampling_rate=sampling_rate)

    # segment
    rpeaks, = peakdetector(
        signal=filtered, sampling_rate=sampling_rate, **peak_params)

    # correct R-peak locations
    rpeaks, = correct_rpeaks(
        signal=filtered, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)

    # extract templates
    templates, rpeaks = extract_heartbeats(
        signal=filtered,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        before=0.2,
        after=0.4)

    # compute heart rate
    try:
        hr_idx, hr = st.get_heart_rate(
            beats=rpeaks, sampling_rate=sampling_rate, smooth=True, size=3)
    except Exception:
        hr_idx = [0, 1]
        hr = [0, 0]
        # print(templates.shape, templates)
        templates = np.append(templates, np.zeros(180))
        templates = templates.reshape(-1, 180)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=False)
    ts_hr = ts[hr_idx]
    ts_tmpl = np.linspace(-0.2, 0.4, templates.shape[1], endpoint=False)

    # plot
    if show:
        plotting.plot_ecg(
            ts=ts,
            raw=signal,
            filtered=filtered,
            rpeaks=rpeaks,
            templates_ts=ts_tmpl,
            templates=templates,
            heart_rate_ts=ts_hr,
            heart_rate=hr,
            path=None,
            show=True)

    # output
    args = (ts, filtered, rpeaks, ts_tmpl, templates, ts_hr, hr)
    names = ('ts', 'filtered', 'rpeaks', 'templates_ts', 'templates',
             'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def extract_data(biooutput, sampling_rate=300):
    original_signal = biooutput["original"]
    filtered_signal = biooutput["filtered"]
    median_template = np.median(biooutput["templates"], axis=0)
    # plt.plot(median_template)
    mean_template = np.mean(biooutput["templates"], axis=0)
    std_template = np.std(biooutput["templates"], axis=0)
    # plt.plot(std_template)
    # plt.show()
    if len(median_template) == 0:
        median_template = np.zeros((0.6 * sampling_rate))
        mean_template = np.zeros((0.6 * sampling_rate))
        std_template = np.zeros((0.6 * sampling_rate))

    if len(biooutput["heart_rate"]) == 0:
        heart_rate = [0]
    else:
        heart_rate = biooutput["heart_rate"]

    if len(biooutput["rpeaks"]) == 0:
        rpeaks = [0]
    else:
        rpeaks = biooutput["rpeaks"]

    median_template_stats = signal_stats(median_template)
    mean_template_stats = signal_stats(mean_template)
    heartrate_stats = signal_stats(heart_rate)

    peak_values = original_signal[rpeaks]
    peak_stats = signal_stats(peak_values)

    heartrate_percentiles = np.percentile(heart_rate,
                                          [5, 15, 25, 35, 65, 75, 85, 95])
    peaks_percentiles = np.percentile(rpeaks, [5, 15, 25, 35, 65, 75, 85, 95])
    median_template_percentiles = np.percentile(
        median_template, [5, 15, 25, 35, 65, 75, 85, 95])
    mean_template_percentiles = np.percentile(mean_template,
                                              [5, 15, 25, 35, 65, 75, 85, 95])

    return (filtered_signal, median_template, mean_template, std_template,
            heartrate_percentiles, peaks_percentiles,
            median_template_percentiles, mean_template_percentiles,
            median_template_stats, mean_template_stats, heartrate_stats,
            peak_stats)


def featurevector(processed_signal, sampling_rate=300):
    results = extract_data(processed_signal, sampling_rate=sampling_rate)
    filtered_signal = results[0]
    median_template = results[1]
    mean_template = results[2]
    std_template = results[3]
    heartrate_percentiles = results[4]
    peaks_percentiles = results[5]
    median_template_percentiles = results[6]
    mean_template_percentiles = results[7]
    median_template_stats = list(results[8].as_dict().values())
    mean_template_stats = list(results[9].as_dict().values())
    heartrate_stats = list(results[10].as_dict().values())
    peak_stats = list(results[11].as_dict().values())

    features = np.array([])
    features = np.append(features, median_template)
    features = np.append(features, mean_template)
    features = np.append(features, std_template)
    features = np.append(features, heartrate_percentiles)
    # features = np.append(features, peaks_percentiles)
    # features = np.append(features, median_template_percentiles)
    # features = np.append(features, mean_template_percentiles)
    features = np.append(features, median_template_stats)
    features = np.append(features, mean_template_stats)
    features = np.append(features, heartrate_stats)
    features = np.append(features, peak_stats)

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
