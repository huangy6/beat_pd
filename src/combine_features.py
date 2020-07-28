import pandas as pd
import numpy as np


def combine_features(feature_files):
    print("TODO: combine_features.py")


if __name__ == "__main__":
    combined_df = combine_features(snakemake.input)
    combined_df.to_csv(snakemake.output[0])