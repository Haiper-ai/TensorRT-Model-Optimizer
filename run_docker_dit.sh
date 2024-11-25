#!/usr/bin/env bash

# use realesr instead of gcr.io/tputryout/deepspeed_inference:latest for inference

# Torch 2.4 docker:
#   gcr.io/tputryout/deepspeed_train:nv2405_v3_compiled
#   gcr.io/tputryout/deepspeed_train:nv2405_v3_compiled_new_diffusers
#   gcr.io/tputryout/deepspeed_train:nv2405_v3_compiled_new_diffusers_2 (default)
# Torch 2.3 docker:
#   gcr.io/tputryout/deepspeed_train:pt231_cu118_ds0144_compiled
#   gcr.io/tputryout/deepspeed_train:pt231_cu118_ds0144_compiled_new_diffusers
# Older Torch docker:
#   gcr.io/tputryout/deepspeed_train:m120_cu118_ds0142_xf025_fa257_compiled
# RIFE docker:
#   gcr.io/tputryout/deepspeed_train:cog5b_rife

# add option for docker image
DOCKER_IMG="docker.io/library/modelopt_examples:latest"
# parse input arguments, -i for docker image, -h for help
while getopts "i:r:h" opt; do
  case $opt in
    i) DOCKER_IMG=$OPTARG ;;
    r) DOCKER_NAME=$OPTARG ;;
    h) echo "Usage: run_docker.sh [-i DOCKER_IMG] [-r DOCKER_NAME]"; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
  esac
done

echo "Using docker image: $DOCKER_IMG"

docker run -v "$(pwd)":/workspace/TensorRT-Model-Optimizer \
           -v /data1/ymiao/ais:/ais_data \
           -v /data1/models/:/data1/models/ \
           -w /workspace/TensorRT-Model-Optimizer \
           --shm-size 32G \
           --rm --gpus 'all' \
           --ipc=host \
           --network=host \
           --name="cog_quantize" \
           -u 0 -t -i $DOCKER_IMG bash

# Use --gpus '"device=5,6,7"' to specify which GPUs to use.
# --add-host=host.docker.internal:host-gateway \
