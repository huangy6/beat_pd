import pandas as pd
import numpy as np

from constants import *


def predict_by_measurement(model, measurement_df):
    print("# TODO: fill in src/predict_by_measurement.py")

    prediction_df = None

    return prediction_df


# The code below should need little to no modifications for other teams.
if __name__ == "__main__":
    # Read the measurement CSV into a data frame.
    measurement_df = pd.read_csv(snakemake.input['measurement'])
    # Load the model.
    model = pd.read_csv(snakemake.input['model']) # TODO(Mark): change this
    # Make a prediction based on this model and measurement data frame.
    prediction_df = predict_by_measurement(model, measurement_df)
    # Save the prediction information to a CSV file.
    prediction_df.to_csv(snakemake.output[0])