#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python main.py --train True --test False
CUDA_VISIBLE_DEVICES=1 python main.py --train False --test True
