#!/bin/bash

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=2235 --max_restarts=0 eval.py --config-name=virtue_eval