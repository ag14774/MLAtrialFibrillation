import sys

import numpy as np
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelmax
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
        args = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        names = ('mean', 'median', 'maxabs', 'minabs', 'max', 'min', 'var',
                 'std_dev', 'abs_dev', 'kurtosis', 'skewness')

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


def qrs_duration(templates, sampling_rate=300):
    rpeak = round(0.2 * sampling_rate)
    before = round(0.08 * sampling_rate)
    after = round(0.08 * sampling_rate)
    qrs_durations = []
    for t in templates:
        start = np.argmin(t[rpeak - before:rpeak])
        start = rpeak - before + start
        end = np.argmin(t[rpeak:rpeak + after])
        end = rpeak + end
        qrs_durations.append((end - start) / sampling_rate)
    return np.array(qrs_durations)


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

    rrintervals = RRIntervals(rpeaks)
    rrinterval_stats = signal_stats(rrintervals)

    median_temp_perc = np.percentile(median_template,
                                     [5, 15, 25, 35, 65, 75, 85, 95])
    mean_temp_perc = np.percentile(mean_template,
                                   [5, 15, 25, 35, 65, 75, 85, 95])
    heart_rate_perc = np.percentile(heart_rate,
                                    [5, 15, 25, 35, 65, 75, 85, 95])
    peak_perc = np.percentile(peak_values, [5, 15, 25, 35, 65, 75, 85, 95])

    rrinterval2 = np.diff(rrintervals)
    rrinterval2_stats = signal_stats(rrinterval2)

    rr50 = np.sum(rrinterval2 > 0.05)
    rr20 = np.sum(rrinterval2 > 0.02)
    prr50 = rr50 / len(rrinterval2)
    prr20 = rr20 / len(rrinterval2)
    hrv_data = np.array([rr50, rr20, prr50, prr20])

    return (filtered_signal, median_template, mean_template, std_template,
            median_template_stats, mean_template_stats, heartrate_stats,
            peak_stats, rrinterval_stats, median_temp_perc, mean_temp_perc,
            heart_rate_perc, peak_perc, rrinterval2_stats, hrv_data)


@jit
def flatten(lis):
    new_lis = []
    for item in lis:
        if isinstance(item, list):
            new_lis.extend(item)
        else:
            new_lis.append(item)
    return new_lis


def featurevector(processed_signal, sampling_rate=300):
    results = extract_data(processed_signal, sampling_rate=sampling_rate)
    filtered_signal = results[0]
    median_template = results[1]
    mean_template = results[2]
    std_template = results[3]
    median_template_stats = flatten(list(results[4].as_dict().values()))
    mean_template_stats = flatten(list(results[5].as_dict().values()))
    heartrate_stats = flatten(list(results[6].as_dict().values()))
    peak_stats = flatten(list(results[7].as_dict().values()))
    rr_interval_stats = flatten(list(results[8].as_dict().values()))
    median_temp_perc = results[9]
    mean_temp_perc = results[10]
    heart_rate_perc = results[11]
    peak_perc = results[12]
    rrinterval2_stats = results[13]
    hrv_data = results[14]

    features = np.array([])
    features = np.append(features, median_template)
    features = np.append(features, mean_template)
    features = np.append(features, std_template)
    features = np.append(features, median_template_stats)
    features = np.append(features, mean_template_stats)
    features = np.append(features, heartrate_stats)
    features = np.append(features, peak_stats)
    features = np.append(features, rr_interval_stats)
    features = np.append(features, median_temp_perc)
    features = np.append(features, mean_temp_perc)
    features = np.append(features, heart_rate_perc)
    features = np.append(features, peak_perc)
    features = np.append(features, rrinterval2_stats)
    features = np.append(features, hrv_data)

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


def fftanalysis(signal, rate=300, first=8):
    try:
        fftsignal = np.abs(np.fft.fft(signal))**2
        if len(fftsignal < 8):
            fftsignal = np.append(fftsignal, np.zeros(first - len(fftsignal)))
    except Exception:
        fftsignal = np.zeros(first)
    # print(fftsignal)
    # freqs = np.fft.fftfreq(len(signal), d=1 / rate)
    # print(freqs)
    # plt.bar(freqs[:first], fftsignal[:first])
    # plt.show()
    return fftsignal[:first]
