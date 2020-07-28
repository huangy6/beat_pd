import pandas as pd
import numpy as np


def extract_features_by_measurement(measurement_df):
    print("TODO: extract_features_by_measurement.py")


if __name__ == "__main__":
    measurement_df = pd.read_csv(snakemake.input[0])
    features_df = extract_features_by_measurement(measurement_df)
    features_df.to_csv(snakemake.output[0])