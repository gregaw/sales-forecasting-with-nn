#!/usr/bin/env bash

source `dirname $0`/config.txt

if [ ! -d "gcp-output" ]; then
	mkdir gcp-output
fi
# parallel (-m), recursice (-r), no-clobber (-n)
gsutil -m cp -rn ${GCS_BUCKET} gcp-output
cp -R gcp-output/${GCS_BUCKET_NAME} output
