import pandas as pd
import tsfresh as tsf
import argparse

"""
For a particular measurement ID file 
(corresponding to a CSV file in data/cis-pd/training_data)
extract tsfresh features for each window of `window_size` seconds.
Move the window forward by `window_offset` 
until the end of the measurement file is reached.
"""
def extract_tsf_features(input_file, window_offset=5, window_size=10, n_jobs=1):
    df = pd.read_csv(input_file)
    df["Timestamp"] = df["Timestamp"].astype(float)

    timestamp_min = df["Timestamp"].min()
    timestamp_max = df["Timestamp"].max()

    tsf_df = None

    window_start = timestamp_min
    while window_start + window_size < timestamp_max:
        window_stop = window_start + window_size

        window_df = df.loc[(df["Timestamp"] >= window_start) & (df["Timestamp"] < window_stop)]
        window_df = window_df.melt(id_vars=["Timestamp"], value_vars=["X", "Y", "Z"], var_name="dim")
        window_tsf_df = tsf.feature_extraction.extraction.extract_features(window_df, column_id="dim", column_sort="Timestamp", n_jobs=n_jobs)
        window_tsf_df["window_start"] = window_start
        window_tsf_df["window_stop"] = window_stop
        window_tsf_df["window_count"] = df.shape[0]
        window_tsf_df = window_tsf_df.reset_index().rename(columns={"index": "dim"})

        try:
            tsf_df = tsf_df.append(window_tsf_df, ignore_index=True)
        except:
            tsf_df = window_tsf_df

        window_start += window_offset

    return tsf_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, help='Path to a measurement.csv file')
    parser.add_argument('-o', '--output_file', type=str, help='Path for the output CSV file.')
    args = parser.parse_args()
    
    tsf_df = extract_tsf_features(args.input_file)
    tsf_df.to_csv(args.output_file)