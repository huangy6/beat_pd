import json

if __name__ == "__main__":
    # Just manually create the annotations

    dataset_id = snakemake.wildcards['dataset_id']

    print(f"TODO: create annotations for {dataset_id}")

    with open(snakemake.output[0], 'w') as f:
        json.dump({
            "method": "TODO",
            "window_size": 10,
            "overlap": 5,
            "aggregation_strategy": "TODO",
            "resampling_rate": "TODO"
        }, f)