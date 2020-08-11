import json

from constants import F_HYPERPARAMS, F_HYPERPARAM_VALS

if __name__ == "__main__":

    with open(snakemake.output[0], 'w') as f:
        json.dump({
            "method": "ensemble models on windowed time series features",
            "aggregation_strategy": "ensemble prediction by taking median of window predictions",
            "window_size": F_HYPERPARAM_VALS[F_HYPERPARAMS.WINDOW_SIZE.value],
            "overlap": F_HYPERPARAM_VALS[F_HYPERPARAMS.WINDOW_OFFSET.value],
            "resampling_rate": F_HYPERPARAM_VALS[F_HYPERPARAMS.RESAMPLE_RATE.value],
        }, f)