#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="16g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create laibi_algorithm-output-$VOLUME_SUFFIX

# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --gpus 0 \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/input/:/input/ \
        -v laibi_algorithm-output-$VOLUME_SUFFIX:/output/ \
        laibi_algorithm:latest

docker run --rm \
        -v laibi_algorithm-output-$VOLUME_SUFFIX:/output/ \
        bash:5.2 bash -lc "ls -R /output && ls -R /output/images/pancreatic-tumor-segmentation || true"

# Validate that at least one .nii.gz exists under the GC output subdirectory
docker run --rm \
        -v laibi_algorithm-output-$VOLUME_SUFFIX:/output/ \
        bash:5.2 bash -lc "shopt -s nullglob; files=(/output/images/pancreatic-tumor-segmentation/*.nii.gz); if (( \${#files[@]} > 0 )); then exit 0; else exit 1; fi"

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm laibi_algorithm-output-$VOLUME_SUFFIX