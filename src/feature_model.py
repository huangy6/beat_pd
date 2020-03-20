import pandas as pd
import numpy as np

from statsmodels.tsa.arima_model import ARIMA

def extract_arima_features(seq, model_order=(5, 0, 2)):
    model = ARIMA(seq, order=model_order)
    model_fit = model.fit(disp=0, trend='nc')
    return model_fit.params
