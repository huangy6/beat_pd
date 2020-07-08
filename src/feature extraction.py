#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tsfresh as tsf
import glob
import os
import numpy as np
import pandas as pd
import seaborn as sns

from importlib import reload
from datetime import timedelta
from scipy import signal
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from tsfresh.utilities.distribution import ClusterDaskDistributor
from . import main, feature_model, extract_tsf_features_by_window as extract
from dask_jobqueue import SLURMCluster
from distributed import Client, LocalCluster, as_completed


# # Extract features from data

# In[ ]:


# Uncomment for SLURM execution
# cluster_type = SLURMCluster
# cluster_kwargs = {'queue':'short', 'cores':2, 'memory':'4gB', 'walltime':'3:00:00', 'death_timeout':60}
# dapt_kwargs = {'minimum': 1, 'maximum': 100}

# Uncomment for local execution
cluster_type = LocalCluster
cluster_kwargs = {}
adapt_kwargs = {'minimum': 1, 'maximum': 3}


# In[14]:


try:
    cluster.close()
    client.close()
except NameError:
    pass
finally:
    client, cluster = main.choose_dask_cluster(cluster_type, cluster_kwargs, adapt_kwargs)


# In[17]:


client


# In[4]:


cis_colnames = {'t_colname': 'Timestamp', 'xyz_colnames': ['X', 'Y', 'Z']}
smartwatch_colnames = {'devid_colnames': ['device_id']}


# In[5]:


def extract_tsf_features(input_fp, 
                         window_offset=5, 
                         window_size=10, 
                         samp_rate='100ms',
                         rms_g_constant=1,
                         colnames=dict()):
    seq = main.read_seq(input_fp, use_time_index=True, resample=samp_rate, **colnames)
    # some slight interpolation for missing values
    seq = seq.interpolate(axis=0, limit=1, method='linear')
    
    # subtract constant for gravity
    rms = pd.DataFrame({'rms': np.sqrt(np.square(seq).sum(axis=1, skipna=False)) - rms_g_constant})
    
    window_starts = [pd.Timedelta(seconds=t) for t in [*range(0, rms.index.get_level_values('t').max().seconds - window_size, window_offset)]]
    samples = main.sample_seq(rms, starts=window_starts, samp_len=pd.Timedelta(seconds=window_size), reset_time=True)
    for i, df in enumerate(samples):
        df['ord'] = str(i)
        if 'devid_colnames' in colnames:
            df.reset_index(level=colnames['devid_colnames'], inplace=True)
            df['ord'] += '-' + df[colnames['devid_colnames'][0]]
            df.drop(columns=colnames['devid_colnames'], inplace=True)
    
    # remove windows with nulls
    tsf_data = pd.concat(samples, axis=0).groupby('ord').filter(lambda x: x.notnull().values.all())
    
    tsf_df = tsf.extract_features(tsf_data, column_id="ord", disable_progressbar=True, n_jobs=0)
    samp_id = os.path.splitext(os.path.basename(input_fp))[0]
    tsf_df['samp_id'] = samp_id

    return tsf_df


# In[6]:


window_size = 10
window_offset = 5
futures = []


# ## training data

# In[33]:


# Training data for cis_pd
fps = glob.glob('data/cis-pd/training_data/*.csv')
futures = client.map(extract_tsf_features, fps, 
                          window_size=window_size, 
                          window_offset=window_offset, 
                          rms_g_constant=1, 
                          colnames=cis_colnames)


# In[ ]:


# Write to disk directly since too much to store in mem
iterator = as_completed(futures)
future = next(iterator)
while future.status == 'error': 
    future = next(iterator)
result = future.result()
result.to_csv('extracted_features/ensem/cis-tsfeatures.csv', header=True)

# Write remaining dfs in append mode 
for future in tqdm(iterator, total=len(futures)-1):
    if future.status == 'finished':
        result = future.result()
        result.to_csv('extracted_features/ensem/cis-tsfeatures.csv', header=False, mode='a')


# In[ ]:


# real_pd smartphone accelerometer
fps = glob.glob('data/real-pd/training_data/smartphone_accelerometer/*.csv')
futures = client.map(extract_tsf_features, fps, 
                          window_size=window_size, 
                          window_offset=window_offset, 
                          rms_g_constant=9.81)


# In[ ]:


# Write to disk directly since too much to store in mem
iterator = as_completed(futures)
future = next(iterator)
while future.status == 'error': 
    future = next(iterator)
result = future.result()
result.to_csv('extracted_features/ensem/real_phone-tsfeatures.csv', header=True)

# Write remaining dfs in append mode 
for future in tqdm(iterator, total=len(futures)-1):
    if future.status == 'finished':
        result = future.result()
        result.to_csv('extracted_features/ensem/real_phone-tsfeatures.csv', header=False, mode='a')


# In[ ]:


# real_pd smartwatch accelerometer
fps = glob.glob('data/real-pd/training_data/smartwatch_accelerometer/*.csv')
futures = client.map(extract_tsf_features, fps, 
                          window_size=window_size, 
                          window_offset=window_offset, 
                          rms_g_constant=9.81, 
                          colnames=smartwatch_colnames)


# In[ ]:


# Write to disk directly since too much to store in mem
iterator = as_completed(futures)
future = next(iterator)
while future.status == 'error': 
    future = next(iterator)
result = future.result()
result.to_csv('extracted_features/ensem/real_watch_accel-tsfeatures.csv', header=True)

# Write remaining dfs in append mode 
for future in tqdm(iterator, total=len(futures)-1):
    if future.status == 'finished':
        result = future.result()
        result.to_csv('extracted_features/ensem/real_watch_accel-tsfeatures.csv', header=False, mode='a')


# In[ ]:


# real_pd smartwatch gyroscope
fps = glob.glob('data/real-pd/training_data/smartwatch_gyroscope/*.csv')
futures = client.map(extract_tsf_features, fps, 
                          window_size=window_size, 
                          window_offset=window_offset, 
                          rms_g_constant=0, 
                          colnames=smartwatch_colnames)


# In[ ]:


# Write to disk directly since too much to store in mem
iterator = as_completed(futures)
future = next(iterator)
while future.status == 'error': 
    future = next(iterator)
result = future.result()
result.to_csv('extracted_features/ensem/real_watch_gyro-tsfeatures.csv', header=True)

# Write remaining dfs in append mode 
for future in tqdm(iterator, total=len(futures)-1):
    if future.status == 'finished':
        result = future.result()
        result.to_csv('extracted_features/ensem/real_watch_gyro-tsfeatures.csv', header=False, mode='a')


# ## Test data

# In[ ]:


# Test data for cis_pd
fps = glob.glob('data/test_set/cis-pd/testing_data/*.csv')
futures = client.map(extract_tsf_features, fps, 
                          window_size=window_size, 
                          window_offset=window_offset, 
                          rms_g_constant=1, 
                          colnames=cis_colnames)


# In[ ]:


# Write to disk directly since too much to store in mem
iterator = as_completed(futures)
future = next(iterator)
while future.status == 'error': 
    future = next(iterator)
result = future.result()
result.to_csv('extracted_features/ensem/cis_test-tsfeatures.csv', header=True)

# Write remaining dfs in append mode 
for future in tqdm(iterator, total=len(futures)-1):
    if future.status == 'finished':
        result = future.result()
        result.to_csv('extracted_features/ensem/cis_test-tsfeatures.csv', header=False, mode='a')


# In[ ]:


# real_pd smartwatch accelerometer
fps = glob.glob('data/test_set/real-pd/testing_data/smartwatch_accelerometer/*.csv')
futures = client.map(extract_tsf_features, fps, 
                          window_size=window_size, 
                          window_offset=window_offset, 
                          rms_g_constant=9.81, 
                          colnames=smartwatch_colnames)


# In[ ]:


# Write to disk directly since too much to store in mem
iterator = as_completed(futures)
future = next(iterator)
while future.status == 'error': 
    future = next(iterator)
result = future.result()
result.to_csv('extracted_features/ensem/real_watch_accel_test-tsfeatures.csv', header=True)

# Write remaining dfs in append mode 
for future in tqdm(iterator, total=len(futures)-1):
    if future.status == 'finished':
        result = future.result()
        result.to_csv('extracted_features/ensem/real_watch_accel_test-tsfeatures.csv', header=False, mode='a')


# ## To parquet

# In[ ]:


# Convert csv files to parquet
fps = glob.glob('extracted_features/ensem/*-tsfeatures.csv')
for fp in fps:
    df = pd.read_csv(fp)
    df.to_parquet(fp[:-4] + '.parquet')

