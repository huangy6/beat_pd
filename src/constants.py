from enum import Enum

class MANIFEST_COLUMNS(Enum):
    MEASUREMENT_ID = "measurement_id"
    MEASUREMENT_FILE = "measurement_file"
    SUBJECT_ID = "subject_id"
    COHORT = "cohort" # realpd or cispd
    DEVICE = "device" # smartphone or smartwatch
    INSTRUMENT = "instrument" # accelerometer or gyroscope

ALL_MANIFEST_COLUMNS = [
    MANIFEST_COLUMNS.MEASUREMENT_ID.value,
    MANIFEST_COLUMNS.MEASUREMENT_FILE.value,
    MANIFEST_COLUMNS.SUBJECT_ID.value,
    MANIFEST_COLUMNS.COHORT.value,
    MANIFEST_COLUMNS.DEVICE.value,
    MANIFEST_COLUMNS.INSTRUMENT.value,
]

class LABEL_COLUMNS(Enum):
    ON_OFF = "on_off"
    DYSKINESIA = "dyskinesia"
    TREMOR = "tremor"

ALL_LABEL_COLUMNS = [
    LABEL_COLUMNS.ON_OFF.value,
    LABEL_COLUMNS.DYSKINESIA.value,
    LABEL_COLUMNS.TREMOR.value
]

# Standardized measurement column names
class MEASUREMENT_COLUMNS(Enum):
    X = "x"
    Y = "y"
    Z = "z"
    TIMESTAMP = "timestamp"
    DEVICE_ID = "device_id"

# Measurement column names by cohort
COHORT_TO_MEASUREMENT_COLUMN_MAP = {
    "cispd": {
        MEASUREMENT_COLUMNS.X.value: "X",
        MEASUREMENT_COLUMNS.Y.value: "Y",
        MEASUREMENT_COLUMNS.Z.value: "Z",
        MEASUREMENT_COLUMNS.TIMESTAMP.value: "Timestamp",
        MEASUREMENT_COLUMNS.DEVICE_ID.value: None,
    },
    "realpd": {
        MEASUREMENT_COLUMNS.X.value: "x",
        MEASUREMENT_COLUMNS.Y.value: "y",
        MEASUREMENT_COLUMNS.Z.value: "z",
        MEASUREMENT_COLUMNS.TIMESTAMP.value: "t",
        MEASUREMENT_COLUMNS.DEVICE_ID.value: "device_id",
    }
}

class F_HYPERPARAMS(Enum):
    WINDOW_SIZE = "window_size"
    WINDOW_OFFSET = "window_offset"
    RESAMPLE_RATE = "resample_rate"
    RMS_G_CONSTANT = "rms_g_constant"


F_HYPERPARAM_VALS = {
    F_HYPERPARAMS.WINDOW_SIZE.value: 10, # s
    F_HYPERPARAMS.WINDOW_OFFSET.value: 5, # s
    F_HYPERPARAMS.RESAMPLE_RATE.value: 100, # ms
    # Gravity constant term varies by cohort, device, and instrument
    F_HYPERPARAMS.RMS_G_CONSTANT.value: {
        ("cispd", "smartphone", "accelerometer"): 1,
        ("cispd", "smartwatch", "accelerometer"): 1,
        ("realpd", "smartphone", "accelerometer"): 9.81,
        ("realpd", "smartwatch", "accelerometer"): 9.81,
        ("realpd", "smartwatch", "gyroscope"): 0,
    }
}
