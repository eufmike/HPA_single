CONFIG_YAML="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single"
SCRIPT="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/HPA_single"

python $SCRIPT/resnet_multicls.py \
    -c $CONFIG_YAML/config_10k.yaml
