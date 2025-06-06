#!/bin/bash

export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=3

torchrun --standalone --nnodes=1 --nproc_per_node=4 pretrain.py