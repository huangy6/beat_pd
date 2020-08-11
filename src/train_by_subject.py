import pandas as pd
import numpy as np
import seaborn as sns
import dill
import json
from scipy import signal, stats
from sklearn import neighbors, linear_model, ensemble, decomposition
from sklearn import feature_selection, model_selection, metrics, dummy, pipeline, preprocessing, compose
from sklearn.base import clone
from dask_ml.model_selection import RandomizedSearchCV
from itertools import product

from combine_features import combine_features


def train_by_subject(labels_df, features_df, cohort, device, instrument, subject_id, label):

    label_cols = ['on_off', 'dyskinesia', 'tremor', 'subject_id']
    id_cols = ['measurement_id', 'id']

    labels_df["subject_id"] = labels_df["subject_id"].astype(str)
    subj_means = labels_df.groupby('subject_id').mean()

    # These features don't compute for a number of observations
    drop_cols = ['rms__friedrich_coefficients__m_3__r_30__coeff_0',
        'rms__friedrich_coefficients__m_3__r_30__coeff_1',
        'rms__friedrich_coefficients__m_3__r_30__coeff_2',
        'rms__friedrich_coefficients__m_3__r_30__coeff_3',
        'rms__max_langevin_fixed_point__m_3__r_30']
    # These fft features are null for our size of windows
    null_fft_cols = ['rms__fft_coefficient__coeff_%d__attr_"%s"' % (n, s) 
                        for n, s in product(range(51, 100), ['abs', 'angle', 'imag', 'real'])]
    # Sample entropy can take inf which screws with models
    inf_cols = ['rms__sample_entropy']

    df = features_df.drop(columns=[*drop_cols, *null_fft_cols, *inf_cols]).dropna().merge(labels_df, right_on='measurement_id', left_on='measurement_id')
    print('%d rows dropped due to nans in features' % (features_df.shape[0] - df.shape[0]))

    # Model

    ## Model spec
    scaler = preprocessing.RobustScaler(quantile_range=(1, 99))
    scaler_pg = {'scaler__quantile_range': [(.1, 99.9), (.5, 99.5), (1, 99), (5, 95), (10, 90)],}

    # Keep features w/ variance in top x%ile 
    var = lambda X, y: np.var(X, axis=0)
    f_select = feature_selection.SelectPercentile(var, percentile=95)
    f_select_pg = {'f_select__percentile': stats.uniform(0, 100)}

    model = ensemble.RandomForestRegressor()
    model_pg = {
        'model__regressor__n_estimators': stats.randint(50, 100),
        'model__regressor__max_depth': stats.randint(10, 25),
        'model__regressor__max_features': [.25, 'auto']
    }

    clip_out = preprocessing.FunctionTransformer(np.clip, kw_args={'a_min': 0, 'a_max': 4})
    clipped_model = compose.TransformedTargetRegressor(regressor=model, inverse_func=clip_out.transform)

    pipe = pipeline.Pipeline([
        ('scaler', scaler), 
        ('f_select', f_select), 
        ('model', clipped_model),
    ], verbose=1)

    param_grid = {
        **scaler_pg,
        **f_select_pg,
        **model_pg,
    }

    metric = metrics.make_scorer(metrics.mean_squared_error, greater_is_better=False)

    cv = model_selection.StratifiedKFold(shuffle=True)

    ## Model eval

    subj_df = df
    print(f'working on {label}')

    labeled_samps = subj_df.dropna(subset=[label])
    if not labeled_samps.shape[0]: 
        print(f'skipping {label}')
        return None
    
    print(labeled_samps.columns.values.tolist())

    y = subj_df.loc[labeled_samps.index, label].astype('int')
    X = labeled_samps.drop(columns=[*label_cols, *id_cols])

    search = RandomizedSearchCV(pipe, param_grid, n_iter=20, scoring=metric, cv=cv, refit=False)
    cv_fit = search.fit(X, y)
    cv_results_df = pd.DataFrame(cv_fit.cv_results_)

    resultset_json = {
        'cohort': cohort,
        'subject_id': subject_id,
        'model_type': str(type(model).__name__),
        'label': label
    }
    win_params = cv_results_df.loc[cv_results_df.rank_test_score == 1, 'params'].values[0]
    winner = pipe.set_params(**win_params)

    return winner, cv_results_df, resultset_json


if __name__ == "__main__":
    labels_df = pd.read_csv(snakemake.input['labels'])
    feature_files = snakemake.input['features']
    features_df = combine_features(feature_files)

    cohort = snakemake.wildcards['cohort']
    device = snakemake.wildcards['device']
    instrument = snakemake.wildcards['instrument']
    subject_id = snakemake.wildcards['subject_id']

    label = snakemake.wildcards['label']

    winner, cv_results_df, resultset_json = train_by_subject(
        labels_df, 
        features_df, 
        cohort, 
        device, 
        instrument, 
        subject_id, 
        label
    )
    with open(snakemake.output['model'], 'wb') as f:
        dill.dump(winner, f)
    
    cv_results_df.to_csv(snakemake.output['cv_results'])

    with open(snakemake.output['model_info'], 'w') as f:
        json.dump(resultset_json, f)