from os.path import join

configfile: 'config.yml'

# Directory / file constants
SRC_DIR = "src"
DATA_DIR = "data"
RAW_DIR = join(DATA_DIR, "raw")
PROCESSED_DIR = join(DATA_DIR, "processed")

FEATURES_DIR = join(PROCESSED_DIR, "features")
MODELS_DIR = join(PROCESSED_DIR, "models")
PREDICTIONS_DIR = join(PROCESSED_DIR, "predictions")
LABELS_DIR = join(PROCESSED_DIR, "labels")

TEST = "test"
TRAIN = "train"

assert('team' in config.keys())
assert(TRAIN in config.keys())
assert(TEST in config.keys())

TRAIN_SETS = config[TRAIN] if type(config[TRAIN]) == list else []
TEST_SETS = config[TEST] if type(config[TEST]) == list else []
TEAM_NAME = config['team']