import pandas as pd
import numpy as np
import dill
import json

from combine_features import combine_features


def train_by_subject(labels_df, features_df, cohort, device, instrument, subject_id, label):
    print("# TODO: fill in src/train_by_subject.py")

    winner = None
    cv_results_df = None
    resultset_json = None

    return winner, cv_results_df, resultset_json


# The code below should need little to no modifications for other teams.
if __name__ == "__main__":
    # Read the labels from the labels CSV file.
    labels_df = pd.read_csv(snakemake.input['labels'])
    # Obtain a list of paths to all feature files for the current subject.
    feature_files = snakemake.input['features']
    # Combine the feature files for the current subject into one data frame.
    features_df = combine_features(feature_files)
    # Obtain the wildcard values (as strings) from the job parameters.
    cohort = snakemake.wildcards['cohort']
    device = snakemake.wildcards['device']
    instrument = snakemake.wildcards['instrument']
    subject_id = snakemake.wildcards['subject_id']
    # The value of the label wildcard will be one of ["dyskinesia", "on_off", "tremor"].
    label = snakemake.wildcards['label']
    # Train the subject-specific model.
    winner, cv_results_df, resultset_json = train_by_subject(
        labels_df,
        features_df,
        cohort,
        device,
        instrument,
        subject_id,
        label
    )
    # Save the model to a dill file.
    with open(snakemake.output['model'], 'wb') as f:
        dill.dump(winner, f)
    # Save the cross-validation results to a CSV file.
    cv_results_df.to_csv(snakemake.output['cv_results'])
    # Save other model metadata to a JSON file.
    with open(snakemake.output['model_info'], 'w') as f:
        json.dump(resultset_json, f)