import glob
import os
import main
import feature_model
import pandas as pd

from tqdm.auto import tqdm 
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')

def process_file(fp):
    sample_vecs = []
    samp_id = os.path.basename(os.path.dirname(fp))
    df = main.read_seq(fp)
    try:
        features = pd.concat([feature_model.extract_arima_features(df[axis], model_order=(3, 1, 2)) for axis in df])
        features['sample_id'] = samp_id
        sample_vecs.append(features)
    except: 
        continue
    return sample_vecs

training_samples = glob.glob('data/cis-pd/training_data/training_samples/*/*.csv')
with Pool(20) as p:
    feature_vecs = [*tqdm(map(process_file, training_samples), total=len(training_samples))]

eatures_df = pd.DataFrame(feature_vecs)
features_df.to_csv('arima_features.csv')
