import pandas as pd
import numpy as np


def combine_features(feature_files):
    df = pd.DataFrame()
    
    for feature_file in feature_files:
        feature_df = pd.read_csv(feature_file)
        df = df.append(feature_df, ignore_index=True)
    
    return df


if __name__ == "__main__":
    combined_df = combine_features(snakemake.input)
    combined_df.to_csv(snakemake.output[0])