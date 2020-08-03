import pandas as pd
import numpy as np


def combine_predictions(prediction_files):
    df = pd.DataFrame()
    
    for prediction_file in prediction_files:
        prediction_df = pd.read_csv(prediction_file)
        df = df.append(prediction_df, ignore_index=True)
    
    return df


if __name__ == "__main__":
    combined_df = combine_predictions(snakemake.input)
    combined_df.to_csv(snakemake.output[0])