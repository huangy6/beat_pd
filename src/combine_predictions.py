import pandas as pd
import numpy as np


def combine_predictions(prediction_files):
    print("TODO: combine_predictions.py")


if __name__ == "__main__":
    combined_df = combine_predictions(snakemake.input)
    combined_df.to_csv(snakemake.output[0])