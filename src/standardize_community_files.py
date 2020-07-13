import pandas as pd
import numpy as np
import os

def standardize_community_files(
    # Inputs
    cis_pd_updrs_csv,
    cis_pd_clinic_tasks_csv,
    real_pd_updrs_csv,
    real_pd_hauser_diary_csv,
    # Outputs
    cis_pd_updrs_manifest_csv,
    cis_pd_clinic_tasks_manifest_csv,
    real_pd_updrs_manifest_csv,
    real_pd_hauser_diary_manifest_csv
):
    # TODO

if __name__ == "__main__":

    standardize_community_files(
        # Inputs
        cis_pd_updrs_csv=snakemake.input["cis_pd_updrs_csv"],
        cis_pd_clinic_tasks_csv=snakemake.input["cis_pd_clinic_tasks_csv"],
        real_pd_updrs_csv=snakemake.input["real_pd_updrs_csv"],
        real_pd_hauser_diary_csv=snakemake.input["real_pd_hauser_diary_csv"],
        # Outputs
        cis_pd_updrs_manifest_csv=snakemake.output["cis_pd_updrs_manifest_csv"],
        cis_pd_clinic_tasks_manifest_csv=snakemake.output["cis_pd_clinic_tasks_manifest_csv"],
        real_pd_updrs_manifest_csv=snakemake.output["real_pd_updrs_manifest_csv"],
        real_pd_hauser_diary_manifest_csv=snakemake.output["real_pd_hauser_diary_manifest_csv"]
    )
