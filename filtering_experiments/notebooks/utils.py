
import os
from os.path import join
import sys
import pandas as pd
import numpy as np
import scipy.signal as signal

def apply3d(arr, func):
    filtered_arr_dim_0 = func(arr.T[0])
    filtered_arr_dim_1 = func(arr.T[1])
    filtered_arr_dim_2 = func(arr.T[2])
    return np.stack((filtered_arr_dim_0, filtered_arr_dim_1, filtered_arr_dim_2), axis=-1)

def filter_signals(arr):
    # ### Filtering
    # According to the procedure described by Reyes-Ortiz et al:
    # - median filter (to remove noise)
    # - 3rd order Butterworth low-pass filter with corner frequency of 20 Hz (to remove noise)
    # - Butterworth low-pass filter with corner frequency of 0.3 Hz (to separate into body acceleration and gravity)

    # Digital filter Wn value must be normalized between 0 and 1, 
    # where 1 is the Nyquist frequency (half the sampling rate, which in our case is 50 Hz divided by 2).
    butter1 = signal.butter(3, 20/(50/2), 'lowpass', analog=False, output='ba')
    butter2 = signal.butter(3, 0.3/(50/2), 'lowpass', analog=False, output='ba')

    medfilt_arr = apply3d(arr, lambda x: signal.medfilt(x))
    butter1_arr = apply3d(medfilt_arr, lambda x: signal.lfilter(b=butter1[0], a=butter1[1], x=x, axis=0))
    butter2_arr = apply3d(butter1_arr, lambda x: signal.lfilter(b=butter2[0], a=butter2[1], x=x, axis=0))
    return butter2_arr


def signals_to_features(arr):
    butter2_arr = filter_signals(arr)

    # ### FFT
    # According to the procedure described by Kobayashi et al:
    # - apply FFT along each of the 3 dimensions, to obtain complex matrix $F$
    # - create the auto-correlation matrix $R = F^{*} F$ where $F^{*}$ is the complex conjugate transpose of $F$
    # - take the absolute values of each element in $R$ to obtain $\bar{R}$
    # - unroll the upper triangle of $\bar{R}$ into a vector, denoted $z$
    # - take log of $z$

    F = apply3d(butter2_arr, lambda x: np.fft.fft(x))
    F_star = np.conj(F)
    R = F_star @ F.T
    R_bar = np.abs(R)

    triu_i = np.triu_indices(R_bar.shape[0])
    z = R_bar[triu_i]
    log_z = np.log(z)

    # feature engineering: more of an art than a science?
    peaks, _ = signal.find_peaks(log_z, distance=4000)
    peak_midpoints = (peaks[1:] + peaks[:-1]) / 2
    
    x = np.floor(np.sort(np.concatenate((peaks, peak_midpoints)))).astype(int)
    y = log_z[x]
    peaks_arr = np.stack((x, y), axis=-1)
    peaks_arr_truncated = peaks_arr[0:1500]

    features_df = pd.DataFrame(data=peaks_arr_truncated).rename(columns={ 0: 'freq', 1: 'val' })
    return features_df


if __name__ == "__main__":
    in_df = pd.read_csv(snakemake.input[0], sep=' ', header=None)
    out_df = signals_to_features(in_df.values)
    out_df.to_csv(snakemake.output[0], sep='\t', index=False)
