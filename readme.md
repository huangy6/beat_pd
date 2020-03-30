# Beat-PD challenge 
https://www.synapse.org/#!Synapse:syn20825169

## Using snakemake

Create and activate the conda environment:

```
conda env create -f mark.yml
conda activate beat-pd-mark-env
```

Copy the cluster profile to your `.config` directory:

```
mkdir -p ~/.config/snakemake/beat_pd
cp ./snakemake_profile.yaml ~/.config/snakemake/beat_pd/config.yaml
```

Run snakemake to submit jobs using `sbatch`:

```
snakemake --snakefile tsf.smk --profile beat_pd
```

