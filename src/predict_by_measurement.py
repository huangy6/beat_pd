import pandas as pd
import numpy as np


def predict_by_measurement(model, measurement_df):
    print("TODO: predict_by_measurement.py")


if __name__ == "__main__":
    measurement_df = pd.read_csv(snakemake.input['measurement'])
    model = pd.read_csv(snakemake.input['model']) # TODO change this
    prediction_df = predict_by_measurement(model, measurement_df)
    prediction_df.to_csv(snakemake.output[0])