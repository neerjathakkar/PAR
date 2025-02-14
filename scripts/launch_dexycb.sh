#!/bin/bash

# Check if the GPU number is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <GPU_NUMBER>"
  exit 1
fi

# Default values for booleans and string argument
ROT_ONLY=false
TX_ONLY=false

# Set the GPU number using the first command-line argument
GPU_NUMBER=$1
ROT_ONLY=$2
TX_ONLY=$3

# Convert booleans to true/false if using 1/0
if [ "$ROT_ONLY" = "1" ]; then ROT_ONLY=true; elif [ "$ROT_ONLY" = "0" ]; then ROT_ONLY=false; fi
if [ "$TX_ONLY" = "1" ]; then TX_ONLY=true; elif [ "$TX_ONLY" = "0" ]; then TX_ONLY=false; fi

# assert that rot_only and tx_only are not both true or both false
if [ "$ROT_ONLY" = true ] && [ "$TX_ONLY" = true ]; then
  echo "ROT_ONLY and TX_ONLY cannot both be true"
  exit 1
fi

if [ "$ROT_ONLY" = false ] && [ "$TX_ONLY" = false ]; then
  echo "ROT_ONLY and TX_ONLY cannot both be false"
  exit 1
fi

# Print parsed values for debugging
echo "Running on GPU $GPU_NUMBER"
echo "ROTATION ONLY: $ROT_ONLY"
echo "TX ONLY: $TX_ONLY"

# Set the task name based on rotation or translation only
if [ "$ROT_ONLY" = true ]; then
    TASK_NAME="dexycb_rot"
elif [ "$TX_ONLY" = true ]; then
    TASK_NAME="dexycb_tx"
fi

echo "TASK NAME: $TASK_NAME"


# Set CUDA_VISIBLE_DEVICES to the provided GPU number
export CUDA_VISIBLE_DEVICES=$GPU_NUMBER

if [ "$ROT_ONLY" = true ]; then
  python -m PAR.train -m --config-name dexycb_rotation.yaml task_name="$TASK_NAME" trainer=ddp_unused trainer.devices=1 trainer.num_nodes=1 trainer.max_epochs=1000
fi

if [ "$TX_ONLY" = true ]; then
  python -m PAR.train -m --config-name dexycb_translation.yaml task_name="$TASK_NAME" trainer=ddp_unused trainer.devices=1 trainer.num_nodes=1 trainer.max_epochs=1000
fi