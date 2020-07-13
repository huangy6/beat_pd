from os.path import join
import pandas as pd

include: "featurize.smk"

# Required output files
# Reference: https://www.synapse.org/#!Synapse:syn22152015/wiki/604349

# {teamname}_CIS-PD_Clinic_Tasks.csv // CIS-PD Clinic Tasks Smartwatch Segments
# {teamname}_CIS-PD_UPDRS.csv // CIS-PD UPDRS Smartwatch Segments
# {teamname}_CIS-PD_Hauser_Diaries.csv // REAL-PD Hauser Diary Smartphone and Smartwatch Segments
# {teamname}_REAL-PD_UPDRS.csv // REAL-PD UPDRS Smartphone and Smartwatch Segments

# File format: CSV
# columns: measurement_id, prediction

# Rules
rule predict_all:
    input:
        expand(
            join(PREDICTIONS_DIR, "{dataset_id}", f"{TEAM_NAME}_predictions.csv"),
            dataset_id=DATASET_IDS
        )


rule join_subject_predictions:
    input:
        # TODO: all subjects for this dataset
    output:
        join(PREDICTIONS_DIR, "{dataset_id}", f"{TEAM_NAME}_predictions.csv")


rule predict_by_subject:
    input:
        join(MODELS_DIR, "{dataset_id}", "{cohort}_{subject_id}.model")
    output:
        join(PREDICTIONS_DIR, "{dataset_id}", "{cohort}_{subject_id}.csv")


rule build_model_by_subject:
    input:
        # TODO: all features for this subject
    output:
        join(MODELS_DIR, "{dataset_id}", "{cohort}_{subject_id}.model")

        
