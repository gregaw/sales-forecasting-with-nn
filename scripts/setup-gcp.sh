#!/usr/bin/env bash

source `dirname $0`/config.txt

gsutil mb ${GCS_BUCKET}

gsutil cp data/train1.csv ${GCS_TRAIN_FILE}
gsutil cp data/eval1.csv ${GCS_EVAL_FILE}
