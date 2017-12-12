#!/usr/bin/env bash

source `dirname $0`/config.txt

gsutil -m rm -r $(gsutil ls ${GCS_BUCKET})
gsutil rb ${GCS_BUCKET}