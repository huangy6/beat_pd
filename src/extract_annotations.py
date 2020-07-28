import json


if __name__ == "__main__":

    print("TODO: update annotations")

    with open(snakemake.output[0], 'w') as f:
        json.dump({
            "method": "TODO",
            "window_size": 10,
            "overlap": 5,
            "aggregation_strategy": "TODO",
            "resampling_rate": "TODO"
        }, f)