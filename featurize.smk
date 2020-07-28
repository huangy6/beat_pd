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
def get_all_feature_files(w):
    # Get the list of measurement files from the manifest.csv at the checkpoint.
    manifest_file = join(RAW_DIR, w.split, "manifest.csv")
    manifest_df = pd.read_csv(manifest_file)
    measurement_files = manifest_df["measurement_file"].values.tolist()
    # The feature files have the same names as the measurement files, but they should be in FEATURES_DIR.
    feature_files = [ join(FEATURES_DIR, w.split, m_file) for m_file in measurement_files ]
    return feature_files

# Rules
rule featurize_all:
    input:
        expand(join(FEATURES_DIR, "{split}", "features.csv"), split=SPLITS),
        join(FEATURES_DIR, "annotations.json"),


# Combine extracted measurement feature files into dataset-level features files.
rule combine_features:
    input:
        get_all_feature_files
    output:
        join(FEATURES_DIR, "{split}", "features.csv"),
    script:
        join(SRC_DIR, "combine_features.py")


# Extract features for a single measurement file.
rule extract_features_by_measurement:
    input:
        join(RAW_DIR, "{split}", "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv")
    output:
        join(FEATURES_DIR, "{split}", "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv"),
    script:
        join(SRC_DIR, "extract_features_by_measurement.py")


# Create the dataset-level feature set annotations file.
rule extract_annotations:
    output:
        join(FEATURES_DIR, "annotations.json"),
    script:
        join(SRC_DIR, "extract_annotations.py")

        