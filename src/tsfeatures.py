import tsfresh as tsf
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns

from importlib import reload
from scipy import signal
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from tsfresh.utilities.distribution import ClusterDaskDistributor
import main, feature_model, mp_profiler

seqs = []
samp_rate = '100ms'

# mp_profiler.init_yappi()
distributor = ClusterDaskDistributor(address='10.120.16.104:53264')

for fp in tqdm(glob.glob('data/cis-pd/training_data/*.csv')[:1000]):
    samp_id = os.path.splitext(os.path.basename(fp))[0]
    seq = main.read_seq(fp, t_colname='Timestamp', xyz_colnames=['X', 'Y', 'Z'], use_time_index=True, resample=samp_rate)
    rms = np.sqrt(np.sum(np.square(seq), axis=1))
    df = pd.DataFrame(data={'rms': rms})
    df['id'] = samp_id
    seqs.append(df)

df = pd.concat(seqs)
print(f'resampled at {samp_rate}, sequences have {df.id.value_counts().mean()} samples on average')
results = tsf.extract_features(df, column_id='id', distributor=distributor)
results.to_csv('tsfeatures.csv')
