import os
from os.path import join
import pandas as pd

include: 'constants.smk'

# Required output files
# Reference: https://www.synapse.org/#!Synapse:syn22152015/wiki/604455

# {teamname}_features.csv
# Please include all data for both cohorts, and training/test, in one file if possible

# File format: CSV
# columns: measurement_id, metadata_start, metadata_stop, feature[FEATURENAME1], feature[FEATURENAME2],...

# Helper functions
def dataset_to_feature_files(w):
    # Get the list of measurement files from the manifest.csv at the checkpoint.
    manifest_file = join(RAW_DIR, w.split, w.dataset_id, "manifest.csv")
    manifest_df = pd.read_csv(manifest_file)
    measurement_files = manifest_df["measurement_file"].values.tolist()
    # The feature files have the same names as the measurement files, but they should be in FEATURES_DIR.
    feature_files = [ join(FEATURES_DIR, w.split, w.dataset_id, m_file) for m_file in measurement_files ]
    return feature_files

# Rules
rule featurize_all:
    input:
        expand(join(FEATURES_DIR, TEST, "{dataset_id}", "features.csv"), dataset_id=TEST_SETS),
        expand(join(FEATURES_DIR, TRAIN, "{dataset_id}", "features.csv"), dataset_id=TRAIN_SETS),
        expand(join(FEATURES_DIR, TEST, "{dataset_id}", "annotations.json"), dataset_id=TEST_SETS),
        expand(join(FEATURES_DIR, TRAIN, "{dataset_id}", "annotations.json"), dataset_id=TRAIN_SETS)

# Combine extracted measurement feature files into dataset-level features files.
rule combine_features:
    input:
        dataset_to_feature_files
    output:
        join(FEATURES_DIR, "{split}", "{dataset_id}", "features.csv"),
    script:
        join(SRC_DIR, "join_dataset_features.py")


# Extract features for a single measurement file.
rule extract_features_by_measurement:
    input:
        join(RAW_DIR, "{split}", "{dataset_id}", "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv")
    output:
        join(FEATURES_DIR, "{split}", "{dataset_id}", "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv"),
    script:
        join(SRC_DIR, "extract_features_by_measurement.py")


# Create the dataset-level feature set annotations file.
rule extract_annotations:
    output:
        join(FEATURES_DIR, "{split}", "{dataset_id}", "annotations.json"),
    script:
        join(SRC_DIR, "create_feature_set_annotations.py")

        