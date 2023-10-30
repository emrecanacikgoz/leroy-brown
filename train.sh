#!/bin/bash
#SBATCH --job-name=train-bc
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
##SBATCH --constraint=rtx_a6000
#SBATCH --constraint=tesla_t4
#SBATCH --mem=32G
#SBATCH --time=7-0:0:0
#SBATCH --output=%J.log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80
#SBATCH --mail-user=eacikgoz17@ku.edu.tr

echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a
echo

module load anaconda/3.6
source activate lb
echo 'number of processors:'$(nproc)
nvidia-smi


python run_language_only_policy.py \
    experiment=bc/train \
    name=check-data-dists-ver4\
    training.num_epochs=2000 \
    training.batch_size=16 \
    optimizer.lr=1e-4 \
    optimizer.weight_decay=0.1 \
    
source deactivate

