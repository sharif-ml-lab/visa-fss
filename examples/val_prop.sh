#!/bin/bash
# train a model to segment abdominal CT
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
# CPT="sslreg_alpnet_1q_(0.5geom-0.5reg)"
CPT="val_prop"
DATASET='SABS_Superpix'
NWORKER=4

ALL_EV=(0) # 5-fold cross validation (0, 1, 2, 3, 4)
ALL_SCALE=( "MIDDLE") # config of pseudolabels

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,6]' 

###### Training configs (irrelavent in testing) ######
NSTEP=100_000
DECAY=0.98

MAX_ITER=1000 # defines the size of an epoch
SNAPSHOT_INTERVAL=10000 # interval for saving snapshot
SEED='1234'

NPART=3

TST_REFINE=None

###### Validation configs ######
SUPP_ID='[6]' # using the additionally loaded scan as support

echo ===================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do

    OUTPUT_FILE="./exps/${CPT}_seed_${SEED}/$(date +"%Y-%m-%d-%T").txt"
    PREFIX="proptest_${DATASET}_${NPART}parts_lbgroup${LABEL_SETS}_scale_${SUPERPIX_SCALE}_vfold${EVAL_FOLD}"
    LOGDIR="./exps/${CPT}_seed_${SEED}"
    echo $PREFIX 2>&1 | tee $OUTPUT_FILE
    # echo $PREFIX

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    # RELOAD_PATH="exps/finetune_visa_fss/mt_train_SABS_Superpix_lbgroup0_scale_MIDDLE_vfold${EVAL_FOLD}_SABS_Superpix_sets_0_1shot/1/snapshots/final.pth" # path to the reloaded model
    RELOAD_PATH="pretrained/feature_extractor/visa_fss_ct_fold${EVAL_FOLD}.pth" # path to the reloaded model

    python3 -u tests/val_prop.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
    reload_model_path=$RELOAD_PATH \
    num_workers=$NWORKER \
    scan_per_load=-1 \
    label_sets=$LABEL_SETS \
    'use_wce=True' \
    exp_prefix=$PREFIX \
    'clsname=grid_proto' \
    n_steps=$NSTEP \
    exclude_cls_list=$EXCLU \
    eval_fold=$EVAL_FOLD \
    dataset=$DATASET \
    proto_grid_size=$PROTO_GRID \
    max_iters_per_load=$MAX_ITER \
    min_fg_data=1 seed=$SEED \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE n_sup_part=$NPART proptst_refinement=$TST_REFINE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID 2>&1 | tee $OUTPUT_FILE
    done
done
