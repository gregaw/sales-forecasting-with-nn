#!/usr/bin/env bash

source `dirname $0`/config.txt

gsutil -m cp -rn ${GCS_BUCKET} gcp-output
cp -R gcp-output output
