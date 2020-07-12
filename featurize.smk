from os.path import join

include: "download.smk"

configfile: "config.yml"


# The team= argument should be supplied at the command line.
assert('team' in config.keys())

# The measurements dict should be supplied via the config.yml file.
assert('measurements' in config.keys())

# An optional dataset= argument can be supplied at the command line
# if only interested in running for one of the measurements dict keys.
assert(config['dataset'] in config['measurements'].keys() if 'dataset' in config.keys() else True)

# Get the names of the data sets ("test", "train", etc.)
# If the --config dataset=test command line override has been provided,
# then restrict to a single dataset of interest.
DATASET_IDS = config['measurements'].keys() if 'dataset' not in config.keys() else [ config['dataset'] ]

# The team name config argument.
TEAM_NAME = config['team']

# Required output files
# Reference: https://www.synapse.org/#!Synapse:syn22152015/wiki/604455

# {teamname}_features.csv
# Please include all data for both cohorts, and training/test, in one file if possible

# File format: CSV
# columns: measurement_id, metadata_start, metadata_stop, feature[FEATURENAME1], feature[FEATURENAME2],...

# File annotations: JSON
# {"method":"Some method", "window_size": 30, "overlap": 10, "aggregation_strategy":"None", "resampling_rate":"None"}


# Rules
rule all:
    input:
        expand(
            join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_features.csv"),
            dataset_id=DATASET_IDS
        ),
        expand(
            join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_annotations.json"),
            dataset_id=DATASET_IDS
        )

rule dataset_extract_features:
    input:
        (lambda w: expand(
            join(FEATURES_DIR, "{{dataset_id}}", "{measurement_id}.csv"),
            measurement_id=config['measurements'][w.dataset_id]
        ))
    output:
        join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_features.csv"),
    script:
        join(SRC_DIR, "join_dataset_features.py")

rule create_feature_set_annotations:
    output:
        join(FEATURES_DIR, "{dataset_id}", f"{TEAM_NAME}_annotations.csv"),
    script:
        join(SRC_DIR, "create_feature_set_annotations.py")

        