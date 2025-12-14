#!/bin/sh

# Simple launcher for main.py with EICL/ICL experiments.
# Usage:
#   sh eicl.sh <GPU_ID>
#

# Run smaller LLMs.
CUDA_VISIBLE_DEVICES="$1" python main.py \
  --auxiliary_model GE \
  --experiment_type EICL \
  --dataset EDOS \
  --models gpt-4o-mini
  #--models Phi-3.5
  #--models Phi-3.5 Llama3.1_8b Mistral-Nemo
  

# Run larger LLMs.
#CUDA_VISIBLE_DEVICES="$1" python main.py \
#  --auxiliary_model EI \
#  --experiment_type EICL \
#  --dataset GE \
#  --models gpt-4o-mini Claude-Haiku
