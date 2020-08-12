import json

from constants import *


if __name__ == "__main__":
    print("# TODO: fill in src/extract_annotations.py")

    with open(snakemake.output[0], 'w') as f:
        json.dump({
            # Fill in the appropriate values for your method below.
            "method": "TODO",
            "aggregation_strategy": "TODO",
            "window_size": "TODO",
            "overlap": "TODO",
            "resampling_rate": "TODO",
        }, f)