# Beat-PD challenge 
https://www.synapse.org/#!Synapse:syn20825169

Yidi Huang, Mark Keller, Mohammed Saqib

This repository contains our submission for the BEAT-PD challenge. Our pipeline is presented as a set of two Jupyter notebooks: `feature extraction.ipynb` and `personalized.ipynb`, respectively handling feature extraction and model fitting/evaluation. The *final_sub* branch (default) contains a cleaned up copy of our code, consisting of the minimum necessary files to reproduce our final pipeline. Other approaches we tried can be found in the other branches. 

## Pipeline overview

* Extract tsfresh features from windowed observations 
  * Computed feature representations are saved to `extracted_features/`
* Randomized hyperparameter search with CV
  * Scores from hyperparameter search are saved to `performance/`
* Re-fit with winning hyperparameters 
  * Fitted models are saved to `models/`
* Predict test data
  * Final test set predictions are saved to `test_predictions/`

The first three steps can be resource intensive. We have included cached versions of their results in this repository so each step can be run in isolation. 

## Prerequisites

* Python3.6+ 
* Anaconda - env.yml contains an anaconda environment specification 
* A SLURM cluster - fitted models can be evaluated without a cluster, but feature extraction and fitting will be greatly accelerated using a cluster, and the hyperparameter search is infeasible on a single computer. 
  * The distributed operations can easily be made to run locally by changing `distributed.SLURMCluster` to `distributed.LocalCluster` in the dask initialization cells at the top of each notebook. 

## Pipeline in detail

### Preconditions
The feature extraction step, first in the pipeline, expects to find the raw data files in the `data/` directory. Specifically, it looks for raw sensor csv files in `data/cis-pd/training_data/*.csv` for CIS-PD and `data/real_pd/training_data/*/*.csv` for REAL-PD. It also expects to find the test set sensor files under `data/test_set/{cis,real}-pd/testing_data/{*.csv,*/*.csv}`. 

### Feature extraction
The `feature extraction.ipynb` notebook iterates through the supplied raw data directories and extracts a collection of feature vectors. For each raw sensor file, it computes a composite signal that is the root mean square of the supplied axes. It then extracts 10s data windows for every 5s of signal, and uses the tsfresh library to compute a feature vector for each window. This process is parallelized over data files using a distributed Dask cluster. The outputs of this step are written to the `extracted_features/ensem/` directory, organized by dataset.

### Hyperparameter search
The hyperparameter search takes place in the `Model eval` h2 heading in `personalized.ipynb`. It can be started by running the notebook as shown through the end of the `Model eval` heading to perform the search for all CIS-PD models, and by replacing `dataset='cis-tsfeatures'` with `dataset='real_watch_accel-tsfeatures'` under the `Load data` h1 heading, and replacing `for subj in cis_subjs` with `for subj in real_subjs` under `Model eval`. The search process can be intensive. The search parameters and cluster configuration shown were feasible given our time and resource constraints, but may not be on a different cluster. The results from the search are saved in `performance/cv_paramsweeps/`

### Re-fit on train data
The best performing hyperparameters for each model were used to initialize a new model, which was fitted on all of the available training data. This requires the data be loaded and the model specified by running `personalized.ipynb` through the end of the `Model spec` heading. The first two cells under `Train final model params` perform the model fits in distributed fashion and serializes the fitted models to be saved under `models/final_fitted`. As saved, the notebook is configured for CIS-PD, but can be changed for REAL-PD by making the changes described in the previous section. 

### Predict test data
The set of models used to generate our final test set predictions are saved in the `models/final_fitted/` directory following an intuitive naming scheme. Additionally, the windowed feature vector representations of the test set data can be generated following the **Feature extraction** section above. These can be loaded in order to reproduce our final test set predictions. This step can be reproduced by first running `personalized.ipynb` through the first cell under `Load data`, then running the cells under `Predictions on test set`. Assuming that the files are found in their expected locations, this step will generate a prediction for each sample using the personalized model for that subject, making a naive guess using the subject-specific mean if the model was unable to make a prediction on that sample. 
