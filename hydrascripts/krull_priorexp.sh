#!/bin/bash -l
# number of nodes and cores 
#PBS -l nodes=1:ppn=1
# memory requirements change this when using more/less experience replay samples
#PBS -l mem=16gb
# max run time
#PBS -l walltime=300:00:00
# output and error files
#PBS -o krull_priorexp.out
#PBS -e krull_priorexp.err
#PBS -N krull_priorexp
#PBS -t 1-10
#PBS -V


module add openblas
cd $HOME
source .bashrc
source activate dqn
cd DQN-tensorflow

# Execute the line matching the array index from file one_command_per_index.list:
cmd=`head -${PBS_ARRAYID} ./hydrascripts/krull_priorexp.list | tail -1`

# Execute the command extracted from the file:
eval $cmd