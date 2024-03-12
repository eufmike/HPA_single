#!/bin/bash

# UGE PARAMETERS
#$ -N hpa_3k
#$ -pe smp 1
##### binding seems to cause jobs not allocated resource. https://jirasd.prd.nibr.novartis.net/servicedesk/customer/portal/22/NXSD-522623
#####$ -binding linear:2
#$ -cwd
#$ -S /bin/bash
#$ -l m_mem_free=32G
#$ -l h_rt=3600
#$ -l gpu_card=1
#$ -j y
#$ -o /dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/qsub_log/output_hpa_3k.txt
#$ -e /dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/qsub_log/errors_hpa_3k.txt
# UGE PARAMETERS END

# make sure that the NSLOTS number here matches '#$ -pe smp N' argument above
# NSLOTS=4
# echo "NSLOTS=$NSLOTS"
# export OMP_NUM_THREADS=$NSLOTS
# export MKL_NUM_THREADS=$NSLOTS
# export OPENBLAS_NUM_THREADS=$NSLOTS
# export NUMEXPR_NUM_THREADS=$NSLOTS
# export NUMEXPR_MAX_THREADS=$NSLOTS
# export VECLIB_MAXIMUM_THREADS=$NSLOTS

  
# Set our environment
# Make sure that we have our modules in the MODULEPATH
export MODULEPATH=/usr/prog/modules/all:/cm/shared/modulefiles:$MODULEPATH
# Purge all modules from your .bashrc and profiles. To make sure that your script can be easily shared with outer users
module purge
 
# SET CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=`/cm/shared/apps/nibri/sciComp/get_gpu_map.sh`
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  
# some fancy logging
START=`date +%s`; 
STARTDATE=`date`;
echo "[INFO] [$START] [$STARTDATE] [$$] [$JOB_ID.$SGE_TASK_ID] Starting the workflow"
echo "[INFO] [$START] [$STARTDATE] [$$] [$JOB_ID.$SGE_TASK_ID] We got the following cores: $CUDA_VISIBLE_DEVICES"
 
# run your workflow
# if your workflow is not CUDA based or it's something really custom, please make sure that you pass the GPU cores numbers and use only these inside of your workflow

# Get sample number from the task id
echo "[INFO] Host Name: `hostname`"
echo ""

PS1=PRETEND_LOGGING_SHELL
source /etc/profile
source ~/.bashrc

echo "[INFO] Command Line:"

echo ">>>START COMMAND OUTPUT"#!/bin/sh

conda activate mspytorch 
echo "Current CONDA Env: $CONDA_PREFIX"
which python

# SCRIPT=$1
# source $SCRIPT
source /dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/run_3k.sh $CUDA_VISIBLE_DEVICES

echo ">>>END COMMAND OUTPUT"

# grab EXITCODE if needed
EXITCODE=$?
  
# some fancy logging
END=`date +%s`;
ENDDATE=`date`;
echo "[INFO] [$END] [$ENDDATE] [$$] [$JOB_ID.$SGE_TASK_ID] Workflow finished with code $EXITCODE"
echo "[INFO] [$END] [$ENDDATE] [$$] [$JOB_ID.$SGE_TASK_ID] Workflow execution time (seconds) : $(( $END-$START ))"
