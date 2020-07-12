from os.path import join
import argparse
import pandas as pd
from synapse_utils import get_syn

#' The following code will download both CIS-PD and REAL-PD datasets and write each dataset
#' to a CSV file in the output_dir directory.

def download_tables_from_synapse(output_dir):
    syn, temp = get_syn(output_dir)

    # CIS-PD UPDRS
    cis_pd_updrs_query = syn.tableQuery("select * from syn22232274")
    cis_updrs_paths = syn.downloadTableColumns(cis_pd_updrs_query, "smartwatch_accelerometer")
    cis_pd_updrs = cis_pd_updrs_query.asDataFrame()
    cis_pd_updrs["path"] = (
            cis_pd_updrs
            .smartwatch_accelerometer
            .astype(str)
            .map(cis_updrs_paths))
    cis_pd_updrs.to_csv(join(output_dir, "cis_pd_updrs.csv"), index=False)

    # CIS-PD Clinic tasks
    cis_pd_clinic_tasks_query = syn.tableQuery("select * from syn22232337")
    cis_clinic_tasks_paths = syn.downloadTableColumns(
            cis_pd_clinic_tasks_query, "smartwatch_accelerometer")
    cis_pd_clinic_tasks = cis_pd_clinic_tasks_query.asDataFrame()
    cis_pd_clinic_tasks["path"] = (
            cis_pd_clinic_tasks
            .smartwatch_accelerometer
            .astype(str)
            .map(cis_clinic_tasks_paths))
    cis_pd_clinic_tasks.to_csv(join(output_dir, "cis_pd_clinic_tasks.csv"), index=False)

    # REAL-PD UPDRS
    real_pd_updrs_query = syn.tableQuery("select * from syn22232279")
    real_pd_updrs = {}
    for phenotype in [
            "smartphone_accelerometer",
            "smartwatch_accelerometer",
            "smartwatch_gyroscope"]:
        table_query = syn.tableQuery(
                "select * from syn22232279 where {} is not null".format(phenotype))
        paths = syn.downloadTableColumns(table_query, phenotype)
        df = table_query.asDataFrame()
        df["{}_path".format(phenotype)] = df[phenotype].astype(str).map(paths)
        real_pd_updrs[phenotype] = df
    real_pd_updrs = (real_pd_updrs["smartphone_accelerometer"]
            .merge(real_pd_updrs["smartwatch_accelerometer"], how="outer")
            .merge(real_pd_updrs["smartwatch_gyroscope"], how = "outer"))
    real_pd_updrs.to_csv(join(output_dir, "real_pd_updrs.csv"), index=False)

    # REAL-PD Hauser Diary
    real_pd_hauser_diary_query = syn.tableQuery("select * from syn22232283")
    real_pd_hauser_diary = {}
    for phenotype in [
            "smartphone_accelerometer",
            "smartwatch_accelerometer",
            "smartwatch_gyroscope"]:
        table_query = syn.tableQuery(
                "select * from syn22232283 where {} is not null".format(phenotype))
        paths = syn.downloadTableColumns(table_query, phenotype)
        df = table_query.asDataFrame()
        df["{}_path".format(phenotype)] = df[phenotype].astype(str).map(paths)
        real_pd_hauser_diary[phenotype] = df
    real_pd_hauser_diary = (real_pd_hauser_diary["smartphone_accelerometer"]
            .merge(real_pd_hauser_diary["smartwatch_accelerometer"], how="outer")
            .merge(real_pd_hauser_diary["smartwatch_gyroscope"], how = "outer"))
    real_pd_hauser_diary.to_csv(join(output_dir, "real_pd_hauser_diary.csv"), index=False)

    temp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, required=True)

    args = parser.parse_args()

    download_tables_from_synapse(
        args.output_dir
    )
