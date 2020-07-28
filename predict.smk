import os
from os.path import join
import pandas as pd

include: 'featurize.smk'

# Required output files
# Reference: https://www.synapse.org/#!Synapse:syn22152015/wiki/604349

# {teamname}_CIS-PD_Clinic_Tasks.csv // CIS-PD Clinic Tasks Smartwatch Segments
# {teamname}_CIS-PD_UPDRS.csv // CIS-PD UPDRS Smartwatch Segments
# {teamname}_CIS-PD_Hauser_Diaries.csv // REAL-PD Hauser Diary Smartphone and Smartwatch Segments
# {teamname}_REAL-PD_UPDRS.csv // REAL-PD UPDRS Smartphone and Smartwatch Segments

# File format: CSV
# columns: measurement_id, prediction

# Helper functions
def dataset_to_prediction_files(w):
    # Get the list of measurement files from the manifest.csv at the checkpoint.
    manifest_file = join(RAW_DIR, TEST, "manifest.csv")
    manifest_df = pd.read_csv(manifest_file)
    measurement_files = manifest_df["measurement_file"].values.tolist()
    # The prediction files have the same names as the measurement files, but they should be in PREDICTIONS_DIR.
    prediction_files = [ join(PREDICTIONS_DIR, TEST, m_file) for m_file in measurement_files ]
    return prediction_files

def dataset_and_subject_to_feature_files(w):
    # Get the list of measurement files from the manifest.csv at the checkpoint.
    manifest_file = join(RAW_DIR, TRAIN, "manifest.csv")
    manifest_df = pd.read_csv(manifest_file)
    subject_df = manifest_df.loc[(manifest_df["subject_id"].astype(str) == w.subject_id) & (manifest_df["cohort"] == w.cohort)]
    measurement_files = subject_df["measurement_file"].values.tolist()
    # The prediction files have the same names as the measurement files, but they should be in PREDICTIONS_DIR.
    feature_files = [ join(FEATURES_DIR, TRAIN, m_file) for m_file in measurement_files ]
    print(feature_files)
    return feature_files

# Rules
rule predict_all:
    input:
        join(PREDICTIONS_DIR, "predictions.csv")

# Combine all predictions for a dataset into a single prediction file.
rule combine_predictions:
    input:
        dataset_to_prediction_files
    output:
        join(PREDICTIONS_DIR, "predictions.csv")
    script:
        join(SRC_DIR, "combine_predictions.py")


rule predict_by_measurement:
    input:
        model=join(MODELS_DIR, "{cohort}_{subject_id}.model"),
        measurement=join(FEATURES_DIR, TEST, "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv"),
    output:
        join(PREDICTIONS_DIR, TEST, "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv"),
    script:
        join(SRC_DIR, "predict_by_measurement.py")


# Train a model per-subject by passing all subject measurements and labels to the training script.
rule train_by_subject:
    input:
        dataset_and_subject_to_feature_files,
        labels=join(RAW_TRAIN_DIR, "labels.csv"),
    output:
        join(MODELS_DIR, "{cohort}_{subject_id}.model")
    script:
        join(SRC_DIR, "train_by_subject.py")

        
