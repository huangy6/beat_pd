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