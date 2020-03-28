import os
from os.path import join
import platform

is_o2 = (platform.system() == "Linux")

YIDI_PROJ_DIR = join(os.sep, "home", "hy180", "projects", "beat_pd") if is_o2 else "."
MARK_PROJ_DIR = join(os.sep, "home", "mk596", "research", "beat_pd") if is_o2 else "."

SRC_DIR = "src"
DATA_DIR = join(YIDI_PROJ_DIR, "data", "cis-pd", "training_data")
TSF_DIR = join(MARK_PROJ_DIR, "data", "cis-pd", "training_data_tsf")

MEASUREMENT_FILES = [ f for f in os.listdir(DATA_DIR) if f.endswith(".csv") ]
TSF_FILES = [ (f[:-4] + ".tsf.csv") for f in MEASUREMENT_FILES ]

rule all:
    input:
        [ join(TSF_DIR, f) for f in TSF_FILES ]

rule extract_tsf_features_by_window:
    input:
        join(DATA_DIR, "{measurement_id}.csv")
    output:
        join(TSF_DIR, "{measurement_id}.tsf.csv")
    params:
        script=join(SRC_DIR, "extract_tsf_features_by_window.py")
    shell:
        """
        python {params.script} \
            -i {input} \
            -o {output}
        """
