# TODO: fix

import pandas as pd
import numpy as np
import os
from os.path import join
from shutil import copyfile

from constants import MANIFEST_COLUMNS, ALL_MANIFEST_COLUMNS

def standardize_community_files(io_paths):
    for (input_file, output_file, cohort) in io_paths:
        input_df = pd.read_csv(input_file, index_col=0)
        input_columns = input_df.columns.values.tolist()

        manifest_df = pd.DataFrame(columns=ALL_MANIFEST_COLUMNS, index=[], data=[])
        output_dir = os.path.dirname(output_file)
        
        def create_manifest_row(measurement_id, row, path_col, device, instrument):
            if path_col in input_columns:
                old_measurement_file_path = row[path_col]
                if pd.notnull(old_measurement_file_path):
                    # If there is "path" then there is only one device, always smartwatch_accelerometer
                    subject_id = row['subject_id']
                    new_measurement_file_name = f"{cohort}_{device}_{instrument}_{subject_id}_{measurement_id}.csv"
                    new_measurement_file_path = join(output_dir, new_measurement_file_name)

                    copyfile(old_measurement_file_path, new_measurement_file_path)

                    return {
                        MANIFEST_COLUMNS.COHORT.value: cohort,
                        MANIFEST_COLUMNS.MEASUREMENT_FILE.value: new_measurement_file_name,
                        MANIFEST_COLUMNS.MEASUREMENT_ID.value: measurement_id,
                        MANIFEST_COLUMNS.SUBJECT_ID.value: subject_id,
                        MANIFEST_COLUMNS.DEVICE.value: device,
                        MANIFEST_COLUMNS.INSTRUMENT.value: instrument,
                    }
            return None
        
        for measurement_id, row in input_df.iterrows():
            # If there is "path" then there is only one device, always smartwatch_accelerometer
            if "path" in input_columns:
                manifest_row = create_manifest_row(measurement_id, row, "path", "smartwatch", "accelerometer")
                if manifest_row != None:
                    manifest_df = manifest_df.append(manifest_row, ignore_index=True)
            else:
                for device in ["smartphone", "smartwatch"]:
                    for instrument in ["accelerometer", "gyroscope"]:
                        device_instrument_path = f"{device}_{instrument}_path"
                        manifest_row = create_manifest_row(measurement_id, row, device_instrument_path, device, instrument)
                        if manifest_row != None:
                            manifest_df = manifest_df.append(manifest_row, ignore_index=True)

        manifest_df.to_csv(output_file, index=False)


if __name__ == "__main__":

    standardize_community_files(
        [
            (snakemake.input["cis_pd_updrs_csv"], snakemake.output["cis_pd_updrs_manifest_csv"], "cispd"),
            (snakemake.input["cis_pd_clinic_tasks_csv"], snakemake.output["cis_pd_clinic_tasks_manifest_csv"], "cispd"),
            (snakemake.input["real_pd_updrs_csv"], snakemake.output["real_pd_updrs_manifest_csv"], "realpd"),
            (snakemake.input["real_pd_hauser_diary_csv"], snakemake.output["real_pd_hauser_diary_manifest_csv"], "realpd"),
        ]
    )
