import pandas as pd
import numpy as np


def train_by_subject(labels_df, features_df):
    print("TODO: train_by_subject.py")


if __name__ == "__main__":
    labels_df = pd.read_csv(snakemake.input['labels'])
    feature_files = snakemake.input['features']
    # TODO: read multiple feature files into single features_df

    # TODO: change this, model should not be df/csv
    model_df = train_by_subject(labels_df, features_df)
    model_df.to_csv(snakemake.output[0])