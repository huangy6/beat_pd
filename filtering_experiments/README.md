# accelerometer-data-processing

Attempting to reproduce preprocessing methods discussed in the literature:
- [Rotation invariant feature extraction from 3-D acceleration signals](http://doi.org/10.1109/ICASSP.2011.5947150), Kobayashi et al. IEEE ICASSP 2011
- [Transition-Aware Human Activity Recognition Using Smartphones](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions), Reyes-Ortiz et al. Neurocomputing 2015

## Setup

```sh
conda env create -f environment.yml
conda activate pd-acc-env
```

```sh
snakemake --cores 1
```