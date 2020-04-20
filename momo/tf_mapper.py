import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import lru_cache

@lru_cache(1)
def get_indices():
    labels =  get_labels()
    subjects = labels.groupby("measurement_id").first()["subject_id"]
    train_mid, test_mid = train_test_split(labels.measurement_id.unique(), random_state=1, stratify=subjects)
    subjects = labels[labels.measurement_id.isin(train_mid)].groupby("measurement_id").first()["subject_id"]

    train_mid, valid_mid = train_test_split(train_mid, random_state=1, stratify=subjects)
    all_mid = get_all_mid()
    train_indices  = [all_mid.index(train_m) for train_m in train_mid]
    valid_indices  = [all_mid.index(train_m) for train_m in valid_mid]
    test_indices  = [all_mid.index(train_m) for train_m in test_mid]
    return train_indices, valid_indices, test_indices

@lru_cache(1)
def get_all_mid():
    labels =  get_labels()
    return sorted(labels.measurement_id.unique())

@lru_cache(1)
def get_labels():
    return pd.read_csv("/home/ms994/beat_pd/data/cis-pd/data_labels/CIS-PD_Training_Data_IDs_Labels.csv")

@lru_cache(1)
def get_subjects():
    return sorted(get_labels().subject_id.unique())

@lru_cache(20)
def get_mid_per_subject(subject_id):
    labels = get_labels()
    labels = labels[labels.subject_id==subject_id]
    return sorted(labels.measurement_id)

@lru_cache(20)
def get_splits_per_subject(subject_id):
    train, valid, test = get_indices()
    all_mid = get_all_mid()
    subject_mids = get_mid_per_subject(subject_id)
    subject_mids_ind = [all_mid.index(mid) for mid in subject_mids]
    return set(train).intersection(subject_mids_ind), set(valid).intersection(subject_mids_ind), set(test).intersection(subject_mids_ind),

def read_tfrecord(example):
    features = { \
                'data':  tf.io.FixedLenFeature([1500*3], tf.float32,),\
                'on_off':  tf.io.FixedLenFeature([1], tf.int64,),\
                'dyskinesia':  tf.io.FixedLenFeature([1], tf.int64,),
                'measurement_id':  tf.io.FixedLenFeature([1], tf.int64,),\
                'tremor':  tf.io.FixedLenFeature([1], tf.int64,), \
                'age':  tf.io.FixedLenFeature([1], tf.int64,), \
                "subjects": tf.io.FixedLenFeature([1], tf.int64), \
                "gender": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_PartI_Total": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_PartII_Total": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_4.1": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_4.2": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_4.3": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_4.4": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_4.5": tf.io.FixedLenFeature([1], tf.int64), \
                "UPDRS_4.6": tf.io.FixedLenFeature([1], tf.int64)
               }

    example = tf.io.parse_single_example(example, features)
    return example
def map_example_to_simple(example):
    data = example['data']
    data = tf.reshape(data, (1500,3))
    return data, (example['on_off'][0],)
def tf_is_in_set(a, b):
    return tf.reduce_sum(tf.cast(tf.equal(b, a), tf.int64)) >= 1
#https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/axangles.py
def tfaxangle2mat(x, y, z, angle, is_normalized=False):
#     x, y, z = axis
    if not is_normalized:
        n = tf.math.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    c = tf.math.cos(angle); s = tf.math.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return tf.reshape(tf.concat([
             x*xC+c,   xyC-zs,   zxC+ys ,
             xyC+zs,   y*yC+c,   yzC-xs ,
             zxC-ys,   yzC+xs,   z*zC+c ], axis=-1), (3,3))
std = 1/16 #allow deviation from real rotation with pi/16 std
def map_example_to_simple_train(example):
    data = example['data']
    data = tf.reshape(data, (1500,3))
    update_matrix = tfaxangle2mat(tf.constant(0.0), tf.constant(0.0), tf.constant(1.0), tf.random.normal((1,)) * tf.constant(3.14*std))
    update_matrix = update_matrix @ tfaxangle2mat(tf.constant(0.0), tf.constant(1.0), tf.constant(0.0), tf.random.normal((1,)) * tf.constant(3.14*std))
    update_matrix = update_matrix @ tfaxangle2mat(tf.constant(1.0), tf.constant(0.0), tf.constant(0.0), tf.random.normal((1,)) * tf.constant(3.14*std))
    data = data @ update_matrix
    return data, (example['on_off'][0], example['dyskinesia'][0], example['tremor'][0],)



def map_example_to_simple_no_augment(example):
    data = example['data']
    data = tf.reshape(data, (1500,3))
    return data, (example['on_off'][0], example['dyskinesia'][0], example['tremor'][0],)

def get_batched_dataset(filenames, batch_size, m_ids, max_queue_size=10,  n_process=4, map_example=None, cache=False):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=n_process)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=n_process)
    dataset = dataset.filter(lambda example: tf_is_in_set(example["measurement_id"], tf.constant(m_ids, dtype=tf.int64)))
#     if is_train:
#         dataset = dataset.map(map_example_to_simple_train, num_parallel_calls=n_process)
#     else:
    dataset = dataset.map(map_example, num_parallel_calls=n_process)
    dataset = dataset.filter(lambda x, y: tf.not_equal(y[0], -1))
    dataset = dataset.filter(lambda x, y: tf.math.reduce_any(tf.math.reduce_std(x, axis=0) > 0.05))
    dataset = dataset.filter(lambda x, y: tf.not_equal(y[1], -1))
    dataset = dataset.filter(lambda x, y: tf.not_equal(y[2], -1))

    if cache:
        dataset = dataset.cache()
    dataset = dataset.repeat()
#     if is_train:
    dataset = dataset.shuffle(2056)
    dataset = dataset.batch(batch_size, drop_remainder=True)
#     if is_train:
    dataset = dataset.prefetch(max_queue_size)
#     else:
#         dataset = dataset.prefetch(int(max_queue_size/4)) #store a lot less for the other sets to avoid wasting memory

    return dataset