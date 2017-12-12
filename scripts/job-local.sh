#!/usr/bin/env bash

source `dirname $0`/config.txt

# ------------------------------------ local

#rm -rf ./${JOB_DIR}/*
#python trainer/task.py --train-files $TRAIN_FILE --eval-files $EVAL_FILE --job-dir $JOB_DIR --num-epochs $NUM_EPOCHS --checkpoint-epochs $CHECKPOINT_EPOCHS --eval-frequency $EVAL_FREQUENCY

# ------------------------------------ local-using-gcloud
#

#rm -rf ./${JOB_DIR}/*

gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    -- \
    --train-files ${TRAIN_FILE} \
    --eval-files ${EVAL_FILE} \
    --num-epochs ${NUM_EPOCHS} \
    --job-dir ${JOB_DIR} \
     --checkpoint-epochs ${CHECKPOINT_EPOCHS} \
     --eval-frequency ${EVAL_FREQUENCY}
