# TODO: fix

from os.path import join

# Directory / file constants
# Intended to be used by snakefiles that import this snakefile.
SRC_DIR = "src"
DATA_DIR = "data"
RAW_DIR = join(DATA_DIR, "raw")
PROCESSED_DIR = join(DATA_DIR, "processed")
FEATURES_DIR = join(PROCESSED_DIR, "features")
MODELS_DIR = join(PROCESSED_DIR, "models")
PREDICTIONS_DIR = join(PROCESSED_DIR, "predictions")

# Raw download targets, before cleaning up.
# _Not_ intended to be used by snakefiles that import this snakefile.
DOWNLOAD_DIR = join(RAW_DIR, "download")
DOWNLOAD_CHALLENGE_DIR = join(DOWNLOAD_DIR, "challenge")
DOWNLOAD_COMMUNITY_DIR = join(DOWNLOAD_DIR, "community")

# Rules
checkpoint download_and_standardize:
    output:
        directory(join(RAW_DIR, "{dataset_id}"))

# Standardize the file structure of the challenge-phase files after the initial download.
rule standardize_challenge:
    output:
        directory(join(RAW_DIR, "challenge_cispd")),
        directory(join(RAW_DIR, "challenge_realpd")),
        join(RAW_DIR, "challenge_cispd", "train", "manifest.csv"),
        join(RAW_DIR, "challenge_realpd", "train", "manifest.csv"),
        join(RAW_DIR, "challenge_cispd", "train", "labels.csv"),
        join(RAW_DIR, "challenge_realpd", "train", "labels.csv"),
        join(RAW_DIR, "challenge_cispd", "test", "manifest.csv"),
        join(RAW_DIR, "challenge_realpd", "test", "manifest.csv")
    run:
        print("It appears that this dataset can no longer be accessed via Synapse+BRAIN commons?")


# Standardize the file structure of the community-phase files after the initial download.
rule standardize_community:
    input:
        cis_pd_updrs_csv=join(DOWNLOAD_COMMUNITY_DIR, "cis_pd_updrs.csv"),
        cis_pd_clinic_tasks_csv=join(DOWNLOAD_COMMUNITY_DIR, "cis_pd_clinic_tasks.csv"),
        real_pd_updrs_csv=join(DOWNLOAD_COMMUNITY_DIR, "real_pd_updrs.csv"),
        real_pd_hauser_diary_csv=join(DOWNLOAD_COMMUNITY_DIR, "real_pd_hauser_diary.csv")
    output:
        directory(join(RAW_DIR, "community_cispd_clinic_tasks")),
        directory(join(RAW_DIR, "community_cispd_updrs")),
        directory(join(RAW_DIR, "community_realpd_hauser_diary")),
        directory(join(RAW_DIR, "community_realpd_updrs")),
        cis_pd_updrs_manifest_csv=join(RAW_DIR, "community_cispd_clinic_tasks", "test", "manifest.csv"),
        cis_pd_clinic_tasks_manifest_csv=join(RAW_DIR, "community_cispd_updrs", "test", "manifest.csv"),
        real_pd_updrs_manifest_csv=join(RAW_DIR, "community_realpd_hauser_diary", "test", "manifest.csv"),
        real_pd_hauser_diary_manifest_csv=join(RAW_DIR, "community_realpd_updrs", "test", "manifest.csv")
    script:
        join(SRC_DIR, "standardize_community_files.py")


# Do the initial download of the community-phase clinical files using the synapse script.
rule download_community_clinical:
    output:
        join(DOWNLOAD_COMMUNITY_DIR, "cis_pd_updrs.csv"),
        join(DOWNLOAD_COMMUNITY_DIR, "cis_pd_clinic_tasks.csv"),
        join(DOWNLOAD_COMMUNITY_DIR, "real_pd_updrs.csv"),
        join(DOWNLOAD_COMMUNITY_DIR, "real_pd_hauser_diary.csv")
    params:
        script=join(SRC_DIR, "download_tables_from_synapse.py")
    shell:
        """
        python {params.script} \
            --output_dir {DOWNLOAD_COMMUNITY_DIR}
        """
# TODO: import this snakefile as a subworkflow in featurize.smk and/or predict.smk