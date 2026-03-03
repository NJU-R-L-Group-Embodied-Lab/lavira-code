#!/bin/bash

export GLOG_minloglevel=0
export MAGNUM_LOG=verbose
export EGL_PLATFORM=surfaceless
unset DISPLAY

# Generate timestamp for exp_name and log file
TIMESTAMP=$(date +"%m%d-%H%M%S")

# Create logs directory if it doesn't exist
mkdir -p logs

flag="--exp_name ${TIMESTAMP}
      --run-type eval
      --exp-config vlnce_baselines/config/rxr.yaml
      --nprocesses 1
      --use-rxr-100
      NUM_ENVIRONMENTS 1
      TRAINER_NAME ZS-Evaluator-mp
      TORCH_GPU_IDS [1]
      SIMULATOR_GPU_IDS [1]
      "

echo "Starting experiment: ${TIMESTAMP}-api"
echo "Logging to: logs/${TIMESTAMP}.log"

python run_mp.py $flag 2>&1 | tee logs/${TIMESTAMP}.log