#!/bin/bash -l
# number of nodes and cores 
#PBS -l nodes=1:ppn=1
# memory requirements change this when using more/less experience replay samples
#PBS -l mem=16gb
# max run time
#PBS -l walltime=200:00:00
# output and error files
#PBS -o atl_PRIOR_EXP_DOUBLEQ.out
#PBS -e atl_PRIOR_EXP_DOUBLEQ.err
#PBS -N atl_PRIOR_EXP_DOUBLEQ
#PBS -V

module add openblas
cd $HOME
source .bashrc
source activate dqn
cd DQN-tensorflow
python main.py --env_name=Atlantis-v0 --is_train=True --double_q=True --use_gpu=False  --priority_exp=True
