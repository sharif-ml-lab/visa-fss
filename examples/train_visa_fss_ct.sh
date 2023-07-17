#!/bin/bash
# train a model to segment abdominal CT 
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

####### Shared configs ######
PROTO_GRID=8     # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
NWORKER=4
CPT="visa-fss"
DATASET='SABS_Superpix'

ALL_EV=(2 3 4)                  # 5-fold cross validation (0, 1, 2, 3, 4)
ALL_SCALE=("MIDDLE")        # config of pseudolabels

### Use L/R kidney as testing classes
LABEL_SETS=0 
EXCLU='[]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

### Use Liver and spleen as testing classes
# LABEL_SETS=1 
# EXCLU='[1,6]' 

###### Training configs ######
NSTEP=100_000
VAL_INTERVAL=10_000
DECAY=0.98
QUERY_LENGTH=1
QUERY_RANGE=3
FG_COEFF=0

MAX_ITER=1000                   # defines the size of an epoch
SNAPSHOT_INTERVAL=20_000         # interval for saving snapshot
SEED='2895'
DEVICE='cuda:0'                 # for test

###### Validation configs ######
SUPP_ID='[6]' # using the additionally loaded scan as support

###### SSLReg ######
REFINEMENT=None

for EVAL_FOLD in "${ALL_EV[@]}"
do
    for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
    do

    OUTPUT_FILE="./exps/${CPT}_seed_${SEED}/$(date +"%Y-%m-%d-%T").txt"
    PREFIX="train_${DATASET}_lbgroup${LABEL_SETS}_vfold${EVAL_FOLD}_seed${SEED}"
    LOGDIR="./exps/${CPT}_seed_${SEED}"
    
    # echo $CPT
    # echo $PREFIX
    echo $CPT 2>&1 | tee $OUTPUT_FILE
    echo $PREFIX 2>&1 | tee $OUTPUT_FILE

    if [ ! -d $LOGDIR ]
    then
        mkdir $LOGDIR
    fi

    python3 -u tests/training_visa_fss.py with \
    'modelname=dlfcn_res101' \
    'usealign=True' \
    'optim_type=sgd' \
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
    min_fg_data=1 seed=$SEED device=$DEVICE \
    fg_loss_coeff=$FG_COEFF \
    query_length=$QUERY_LENGTH query_range=$QUERY_RANGE \
    refinement=$REFINEMENT \
    save_snapshot_every=$SNAPSHOT_INTERVAL validate_every=$VAL_INTERVAL \
    superpix_scale=$SUPERPIX_SCALE \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    support_idx=$SUPP_ID 2>&1 | tee $OUTPUT_FILE
    done
done
