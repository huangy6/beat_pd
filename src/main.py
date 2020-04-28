import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn import metrics
from tqdm.auto import tqdm
from pathlib import Path
from numpy import random

RAW_SEQ_DIRS = ['../data/cis-pd/training_data', '../data/real-pd/training_data/smartphone_accelerometer'] #, '../data/real-pd/training_data/smartwatch_accelerometer', '../data/real-pd/training_data/smartwatch_gyroscope']
T_COLNAMES = ['Timestamp', 't'] #, 't', 't']
XYZ_COLNAMES = [['X', 'Y', 'Z'], ['x', 'y', 'z']] #, ['x', 'y', 'z'], ['x', 'y', 'z']]


def sample_seq(seq: pd.DataFrame, n_samples=10, samp_len=pd.Timedelta(seconds=10), starts=None, reset_time=True):
    if starts is None:
        starts = [pd.Timedelta(seconds=t) for t in random.uniform(low=0, high=float(seq.index.max()-samp_len), size=n_samples)]
    idx = pd.IndexSlice
    if type(seq.index) == pd.MultiIndex:
        samples = [seq.xs(idx[start:start+samp_len], level='t', drop_level=False) for start in starts]
    else:
        samples = [seq.loc[start:start+samp_len] for start in starts]
    # Some samples will be empty/incomplete due to lapses in measurements
    # TODO: Address multiple devices in real-pd smartwatch measurements
    if not reset_time:
        return samples
    if type(seq.index) == pd.MultiIndex:
        return [samp.set_index(samp.index.set_levels(samp.index.levels[1] - start, level='t')) for samp,start in zip(samples, starts)]
    else:
        return [samp.set_index(samp.index - start) for samp,start in zip(samples, starts)]

def read_seq(fp: str, t_colname='t', xyz_colnames=['x', 'y', 'z'], devid_colnames=[], use_time_index=False, resample=pd.Timedelta(seconds=(1/50))):
    """ reads a file and returns the associated data

    Parameters
    ----------
    fp : str
        Description of parameter `fp`.
    t_colname : type
        Description of parameter `t_colname`.
    xyz_colnames : type
        Description of parameter `xyz_colnames`.
    use_time_index : bool
        Description of parameter `use_time_index`.
    resample : pd.Timedelta
        how much to resample by. Uses mean resampling

    Returns
    -------
    read_seq(fp: str, t_colname='t', xyz_colnames=['x', 'y', 'z'], use_time_index=False,
        Description of returned object.

    """
    df = pd.read_csv(fp, usecols=[t_colname, *xyz_colnames, *devid_colnames])
    df = df.rename(columns=dict(zip([t_colname, *xyz_colnames], ['t', 'x', 'y', 'z'])))

    time_index = pd.to_timedelta(df['t'], unit="s") if use_time_index else df['t']
    devid_index = [df[c] for c in devid_colnames]
    # Drop explicitly to avoid funny business
    df = df.set_index([*devid_index, time_index,], drop=False).drop(columns=['t', *devid_colnames])
    if use_time_index and resample is not None:
        if devid_colnames:
            df = df.groupby(devid_colnames)
        df = df.resample(resample, level='t').mean()
    return df

def write_seq(seq: pd.DataFrame, fp: str):
    seq.to_csv(fp)

def take_training_samples(n_samples=10, samp_len=10):
    """Given a list of directories containing raw sensor files,
    Take n_samples training samples of samp_len seconds each.
    Place output in training_samples/ directory level with raw files"""
    for raw_dir, t_colname, xyz_colnames in tqdm(zip(RAW_SEQ_DIRS, T_COLNAMES, XYZ_COLNAMES)):
        for fp in tqdm(glob.glob(raw_dir + '/*.csv'), leave=False):
            fn = os.path.basename(fp)
            fd = os.path.dirname(fp)
            out_fd = fd + '/training_samples/' + fn[:-4]
            Path(out_fd).mkdir(parents=True, exist_ok=True)
            seq = read_seq(fp, t_colname, xyz_colnames)
            samples = sample_seq(seq, n_samples, samp_len, reset_time=True)
            for i, sample in enumerate(samples):
                sample.to_csv(f'{out_fd}/{i}.csv')

def plot_performance(y_true, y_pred, title_metric=metrics.mean_squared_error):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.violinplot(data=pd.DataFrame({'actual': y_true, 'predicted': y_pred}), x='actual', y='predicted', ax=ax1)
    
    # confusion matrix
    label_vals = np.sort(y_true.unique())
    cm = metrics.confusion_matrix(y_true, np.round(y_pred), labels=label_vals)
    sns.heatmap(cm, xticklabels=label_vals, yticklabels=label_vals, ax=ax2)
    ax2.set_xlabel('predicted')
    ax2.set_ylabel('actual')
    ax2.invert_yaxis()
    
    score = title_metric(y_true, y_pred)
    _ = fig.suptitle(f'score (default mse): {score}')
