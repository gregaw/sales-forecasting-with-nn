#!/usr/bin/env bash

source `dirname $0`/config.txt

# ------------------------------------ CLOUD

# ML-ENGINE SCALING TIERS
#BASIC	1
#STANDARD_1	10
#PREMIUM_1	75
#BASIC_GPU	3
#
gcloud ml-engine jobs submit training ${JOB_NAME} \
                                 --stream-logs \
                                 --runtime-version 1.2 \
                                 --job-dir ${GCS_JOB_DIR} \
                                 --package-path trainer \
                                 --module-name trainer.task \
                                 --region us-central1 \
                                 --scale-tier BASIC \
                                 -- \
                                 --train-files ${GCS_TRAIN_FILE} \
                                 --eval-files ${GCS_EVAL_FILE} \
                                 --num-epochs ${NUM_EPOCHS} \
                                 --checkpoint-epochs ${CHECKPOINT_EPOCHS} \
                                 --eval-frequency ${EVAL_FREQUENCY}

