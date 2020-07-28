import pandas as pd
import numpy as np
import tsfresh as tsf

from constants import *

def sample_seq(seq: pd.DataFrame, n_samples=10, samp_len=pd.Timedelta(seconds=10), starts=None, reset_time=True):
    if starts is None:
        starts = [pd.Timedelta(seconds=t) for t in random.uniform(low=0, high=float(seq.index.max()-samp_len), size=n_samples)]
    idx = pd.IndexSlice
    if type(seq.index) == pd.MultiIndex:
        samples = [seq.xs(idx[start:start+samp_len], level=MEASUREMENT_COLUMNS.TIMESTAMP.value, drop_level=False) for start in starts]
    else:
        samples = [seq.loc[start:start+samp_len] for start in starts]
    # Some samples will be empty/incomplete due to lapses in measurements
    # TODO: Address multiple devices in real-pd smartwatch measurements
    if not reset_time:
        return samples
    if type(seq.index) == pd.MultiIndex:
        return [samp.set_index(samp.index.set_levels(samp.index.levels[1] - start, level=MEASUREMENT_COLUMNS.TIMESTAMP.value)) for samp,start in zip(samples, starts)]
    else:
        return [samp.set_index(samp.index - start) for samp,start in zip(samples, starts)]

def standardize_measurement_df(df, cohort, use_time_index=False, resample=pd.Timedelta(seconds=(1/50))):
    cohort_has_device_id = (COHORT_TO_MEASUREMENT_COLUMN_MAP[cohort][MEASUREMENT_COLUMNS.DEVICE_ID.value] is not None)
    col_rename_map = { v: k for k, v in COHORT_TO_MEASUREMENT_COLUMN_MAP[cohort].items() if v is not None }
    
    df = df.rename(columns=col_rename_map)
    time_index = pd.to_timedelta(df[MEASUREMENT_COLUMNS.TIMESTAMP.value], unit="s") if use_time_index else df[MEASUREMENT_COLUMNS.TIMESTAMP.value]
    devid_colnames = [ MEASUREMENT_COLUMNS.DEVICE_ID.value ] if cohort_has_device_id else []
    devid_index = [ df[c] for c in devid_colnames ]
    # Drop explicitly to avoid funny business
    df = df.set_index([*devid_index, time_index,], drop=False).drop(columns=[MEASUREMENT_COLUMNS.TIMESTAMP.value, *devid_colnames])
    if use_time_index and resample is not None:
        if cohort_has_device_id:
            df = df.groupby(MEASUREMENT_COLUMNS.DEVICE_ID.value)
        df = df.resample(resample, level=MEASUREMENT_COLUMNS.TIMESTAMP.value).mean()
    return df



def extract_features_by_measurement(measurement_df, cohort, device, instrument, subject_id, measurement_id):
    seq = standardize_measurement_df(measurement_df, cohort, use_time_index=True, resample='100ms')
    # some slight interpolation for missing values
    seq = seq.interpolate(axis=0, limit=1, method='linear')

    # subtract constant for gravity
    rms = pd.DataFrame({ 'rms': np.sqrt(np.square(seq).sum(axis=1, skipna=False)) - RMS_G_CONSTANT })

    # TODO: refactor
    window_offset=5
    window_size=10 
    colnames=dict()

    window_starts = [pd.Timedelta(seconds=t) for t in [*range(0, rms.index.get_level_values(MEASUREMENT_COLUMNS.TIMESTAMP.value).max().seconds - window_size, window_offset)]]
    samples = sample_seq(rms, starts=window_starts, samp_len=pd.Timedelta(seconds=window_size), reset_time=True)
    for i, df in enumerate(samples):
        df['ord'] = str(i)
        if 'devid_colnames' in colnames:
            df.reset_index(level=colnames['devid_colnames'], inplace=True)
            df['ord'] += '-' + df[colnames['devid_colnames'][0]]
            df.drop(columns=colnames['devid_colnames'], inplace=True)
    
    # remove windows with nulls
    tsf_data = pd.concat(samples, axis=0).groupby('ord').filter(lambda x: x.notnull().values.all())
    
    tsf_df = tsf.extract_features(tsf_data, column_id="ord", disable_progressbar=True, n_jobs=0)
    # TODO: is below subject or measurment?
    #samp_id = os.path.splitext(os.path.basename(input_fp))[0]
    #tsf_df['samp_id'] = samp_id

    return tsf_df



if __name__ == "__main__":
    measurement_df = pd.read_csv(snakemake.input[0])
    cohort = snakemake.wildcards['cohort']
    device = snakemake.wildcards['device']
    instrument = snakemake.wildcards['instrument']
    subject_id = snakemake.wildcards['subject_id']
    measurement_id = snakemake.wildcards['measurement_id']
    features_df = extract_features_by_measurement(measurement_df, cohort, device, instrument, subject_id, measurement_id)
    features_df.to_csv(snakemake.output[0])