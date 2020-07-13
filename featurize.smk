from os.path import join
import pandas as pd

include: "download.smk"

assert('team' in config.keys())
assert('datasets' in config.keys())

# Get the names of the data sets of interest ("test", "train", etc.)
# These correspond to subdirectories
# - data/raw/{dataset_id}/
# - data/processed/features/{dataset_id}/
# - etc.
DATASET_IDS = config['datasets']
TEAM_NAME = config['team']

# Required output files
# Reference: https://www.synapse.org/#!Synapse:syn22152015/wiki/604455

# {teamname}_features.csv
# Please include all data for both cohorts, and training/test, in one file if possible

# File format: CSV
# columns: measurement_id, metadata_start, metadata_stop, feature[FEATURENAME1], feature[FEATURENAME2],...

# Helper functions
def dataset_to_feature_files(wildcards):
    # Get the list of measurement files from the manifest.csv at the checkpoint.
    manifest_file = join(checkpoints.download_and_standardize.get(**wildcards).output[0], "manifest.csv")
    manifest_df = pd.read_csv(manifest_file)
    measurement_files = manifest_df["measurement_file"].values.tolist()
    # The feature files have the same names as the measurement files, but they should be in FEATURES_DIR.
    feature_files = [ join(FEATURES_DIR, wildcards.dataset_id, m_file) for m_file in measurement_files ]
    return feature_files

# Rules
rule featurize_all:
    input:
        expand(
            join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_features.csv"),
            dataset_id=DATASET_IDS
        ),
        expand(
            join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_annotations.json"),
            dataset_id=DATASET_IDS
        )


# Combine extracted measurement feature files into one dataset-level features file.
rule join_dataset_features:
    input:
        dataset_to_feature_files
    output:
        join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_features.csv"),
    script:
        join(SRC_DIR, "join_dataset_features.py")


# Extract features for a single measurement file.
rule extract_features_by_measurement:
    input:
        join(RAW_DIR, "{dataset_id}", "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv")
    output:
        join(FEATURES_DIR, "{dataset_id}", "{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv"),
    script:
        join(SRC_DIR, "extract_features_by_measurement.py")


# Create the dataset-level feature set annotations file.
rule create_feature_set_annotations:
    output:
        join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_annotations.json"),
    script:
        join(SRC_DIR, "create_feature_set_annotations.py")

        