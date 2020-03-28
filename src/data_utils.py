import pandas as pd
import numpy as np
import json, os

config = None #global config object. lru cache is smarter tbh, but i like having a globally accessible variable to monkey around with like an idiot



def read_config(path=os.path.realpath("../config.json") else os.environ["CONFIG_PATH"])):
    global config
    if config is None:
        config = json.load(open(path, "rb"))
    return config
def cis_data_dir():
    return read_config()["cis_pd_data_dir"]
def real_pd_data_dir():
    return read_config()["real_pd_data_dir"]
def get_mongo_client():
    '''
    Used for Sacred to record results
    '''
    config = read_config()
    if "mongo_uri" not in config.keys():
        return pymongo.MongoClient()
    else:
        mongo_uri = config["mongo_uri"]
        return pymongo.MongoClient(mongo_uri)

def fft_bin(data, bin, frequency):
    fft_data = np.abs(
                        np.fft.fft(
                            data,
                            axis=0))
    fft_freq = np.fft.fftfreq(data.shape[0], d=1/frequency)
