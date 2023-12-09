#!/bin/sh
#$ -l m_mem_free=16G
#$ -l h_rt=3600
#$ -l gpu_card=2

module load PythonDS
conda info --envs
conda env list
# export _DSDE="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE"
# export CONDA_HOME=$_DSDE/runtime_env/miniconda3
# source /etc/profile &
# source ~/.bashrc & 
# conda activate mspytorch
# python /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/resnet_multicls_multigpu.py -g 0 1