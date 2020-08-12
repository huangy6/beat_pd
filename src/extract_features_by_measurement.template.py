import pandas as pd
import numpy as np

from constants import *


def extract_features_by_measurement(measurement_df, cohort, device, instrument, subject_id, measurement_id):
    print("# TODO: fill in src/extract_features_by_measurement.py")

    features_df = None

    return features_df


# The code below should need little to no modifications for other teams.
if __name__ == "__main__":
    # Read the measurement_df from the input CSV file specified in the featurize.smk snakefile.
    measurement_df = pd.read_csv(snakemake.input[0])
    # Obtain the wildcard values (as strings) from the job parameters.
    cohort = snakemake.wildcards['cohort']
    device = snakemake.wildcards['device']
    instrument = snakemake.wildcards['instrument']
    subject_id = snakemake.wildcards['subject_id']
    measurement_id = snakemake.wildcards['measurement_id']
    # Extract features.
    features_df = extract_features_by_measurement(
        measurement_df,
        cohort,
        device,
        instrument,
        subject_id,
        measurement_id
    )
    # Save extracted features to an output CSV file specified in the featurize.smk snakefile.
    features_df.to_csv(snakemake.output[0], index=True)