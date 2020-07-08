from os.path import join

configfile: 'config.yml'

assert('measurements' in config.keys())
assert(config['dataset'] in config['measurements'].keys() if 'dataset' in config.keys() else True)

# Directory / file constants
SRC_DIR = Path("src")
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"

# Get the names of the data sets ("test", "train", etc.)
# If the --config dataset=test command line override has been provided,
# then restrict to a single dataset of interest.
DATASET_IDS = config['measurements'].keys() if 'dataset' not in config.keys() else [ config['dataset'] ]

# Rules
rule all:
    input:
        expand(
            join(FEATURES_DIR, "{dataset_id}", "summary.json"),
            dataset_id=DATASET_IDS
        )

rule summarize_dataset_features:
    input:
        (lambda w: expand(
            join(FEATURES_DIR, "{{dataset_id}}", "{measurement_id}.csv"),
            measurement_id=config['measurements'][w.dataset_id]
        ))
    output:
        join(FEATURES_DIR, "{dataset_id}", "summary.json")
    script:
        join(SRC_DIR, "join_dataset_features.py")
