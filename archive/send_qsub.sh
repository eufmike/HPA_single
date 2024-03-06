SCRIPT_PATH="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single/run.sh"
qsub \
    -o /dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/qsub/log/output.txt \
    -e /dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/qsub/error/error.txt \
    $SCRIPT_PATH