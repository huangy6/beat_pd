import os
from os.path import join
import platform

is_o2 = (platform.system() == "Linux")

YIDI_PROJ_DIR = join(os.sep, "home", "hy180", "projects", "beat_pd") if is_o2 else "."
MARK_PROJ_DIR = join(os.sep, "home", "mk596", "research", "beat_pd") if is_o2 else "."

SRC_DIR = "src"

TRAIN_DATA_DIR = join(YIDI_PROJ_DIR, "data", "cis-pd", "training_data")
TRAIN_TSF_DIR = join(MARK_PROJ_DIR, "data", "cis-pd", "training_data_tsf")

TEST_DATA_DIR = join(YIDI_PROJ_DIR, "data", "test_set", "cis-pd", "testing_data")
TEST_TSF_DIR = join(MARK_PROJ_DIR, "data", "test_set", "cis-pd", "testing_data_tsf")

TRAIN_TSF_FILES = [ (f[:-4] + ".tsf.csv") for f in os.listdir(TRAIN_DATA_DIR) if f.endswith(".csv") ]
TEST_TSF_FILES = [ (f[:-4] + ".tsf.csv") for f in os.listdir(TEST_DATA_DIR) if f.endswith(".csv") ]

rule all:
    input:
        [ join(TRAIN_TSF_DIR, f) for f in TRAIN_TSF_FILES ],
        [ join(TEST_TSF_DIR, f) for f in TEST_TSF_FILES ]

rule extract_test_tsf_features_by_window:
    input:
        join(TEST_DATA_DIR, "{measurement_id}.csv")
    output:
        join(TEST_TSF_DIR, "{measurement_id}.tsf.csv")
    params:
        script=join(SRC_DIR, "extract_tsf_features_by_window.py")
    shell:
        """
        python {params.script} \
            -i {input} \
            -o {output}
        """

rule extract_train_tsf_features_by_window:
    input:
        join(TRAIN_DATA_DIR, "{measurement_id}.csv")
    output:
        join(TRAIN_TSF_DIR, "{measurement_id}.tsf.csv")
    params:
        script=join(SRC_DIR, "extract_tsf_features_by_window.py")
    shell:
        """
        python {params.script} \
            -i {input} \
            -o {output}
        """
