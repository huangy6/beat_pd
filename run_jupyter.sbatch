#!/bin/bash

#SBATCH --partition=priority
#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH -t 48:00:00
#SBATCH -o notebook_%j.out

set -e

port=${1} # <-----------------

/usr/bin/ssh -N -f -R $port:localhost:$port login01

cat<<EOF
Job server: $(hostname)
Job start: $(date)
Job id: $SLURM_JOB_ID

To connect to this instance run the following command locally and open a tunnel:

  ssh -L $port:localhost:$port $USER@$(hostname)

Instructions to connect to the notebook should appear shortly.

EOF

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)
