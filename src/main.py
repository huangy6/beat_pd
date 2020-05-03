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
    return [samp.set_index(samp.index - start) for samp,start in zip(samples, starts)] if reset_time else samples

def read_seq(fp: str, device_id=None, t_colname='t', devid_colnames=[], xyz_colnames=['x', 'y', 'z'], use_time_index=False, resample=pd.Timedelta(seconds=(1/50))):
    if not reset_time:
        return samples
    if type(seq.index) == pd.MultiIndex:
        return [samp.set_index(samp.index.set_levels(samp.index.levels[1] - start, level='t')) for samp,start in zip(samples, starts)]
    else:
        return [samp.set_index(samp.index - start) for samp,start in zip(samples, starts)]
    df = pd.read_csv(fp)
    if "smartwatch_accelerometer" in fp or "smartwatch_gyroscope" in fp:
        if device_id is None:
            device_id = df.device_id.iloc[0]
        df = df[df.device_id == device_id]
    df = df[[t_colname, *xyz_colnames]]
    df = df.rename(columns=dict(zip([t_colname, *xyz_colnames], ['t', 'x', 'y', 'z'])))
    df = df.set_index('t')
    if use_time_index:
        df = df.set_index(pd.to_timedelta(df.index, unit="s"))
        if resample is not None:
            df = df.resample(resample).mean()
    return df

def get_all_cispd_train_data(m_id):
    return main.read_seq(
        f"/home/ms994/beat_pd/data/cis-pd/training_data/{m_id}.csv",
        t_colname="Timestamp",
        xyz_colnames=["X", "Y", "Z"],
        use_time_index=True,
        resample=pd.Timedelta(seconds=1/25)
    )
def get_cispd_eval_data(m_id):
    return main.read_seq(
        f"{m_id}",
        t_colname="Timestamp",
        xyz_colnames=["X", "Y", "Z"],
        use_time_index=True,
        resample=pd.Timedelta(seconds=1/25)
    )
def get_all_real_pd_test_data(m_id, device_id):
    allData = []
    subdirs = ["smartwatch_accelerometer", "smartwatch_gyroscope", "smartphone_accelerometer"]
    for subdir in subdirs:
        if os.path.isfile(f"/home/ms994/beat_pd/test_set/real-pd/testing_data/{subdir}/{m_id}.csv"):
            allData.append(main.read_seq(
            f"/home/ms994/beat_pd/test_set/real-pd/testing_data/{subdir}/{m_id}.csv",
            device_id=device_id,
            use_time_index=True,
            resample=pd.Timedelta(seconds=1/25)))
        else:
            allData.append(pd.DataFrame(index=pd.timedelta_range(start=pd.Timedelta(seconds=0), freq=pd.Timedelta(seconds=1/25), end = pd.Timedelta(minutes=20)), columns=[1,2,3])) #this is basically null
    return allData
def get_all_real_pd_train_data(m_id, device_id):
    allData = []
    subdirs = ["smartwatch_accelerometer", "smartwatch_gyroscope", "smartphone_accelerometer"]
    for subdir in subdirs:
        if os.path.isfile(f"/home/ms994/beat_pd/data/real-pd/training_data/{subdir}/{m_id}.csv"):
            allData.append(main.read_seq(
            f"/home/ms994/beat_pd/data/real-pd/training_data/{subdir}/{m_id}.csv",
            device_id=device_id,
            use_time_index=True,
            resample=pd.Timedelta(seconds=1/25)))
        else:
            allData.append(pd.DataFrame(index=pd.timedelta_range(start=pd.Timedelta(seconds=0), freq=pd.Timedelta(seconds=1/25), end = pd.Timedelta(minutes=20)), columns=[1,2,3])) #this is basically null
    return allData
def read_mid_and_split_data_into_windows(params, get_data=None, max_window=pd.Timedelta(seconds=60), overlap=pd.Timedelta(seconds=10)):
        data = get_data(*params)

        all_samples = []
        currentIndex = pd.Timedelta(seconds=0)
        while (currentIndex + max_window < data.index.max()):
            all_samples.append(data.loc[currentIndex:currentIndex+max_window].fillna(method="ffill").fillna(method="bfill").values) #some values are NaNs in the data!
            currentIndex += overlap
        return all_samples

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
