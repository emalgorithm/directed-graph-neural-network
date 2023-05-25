#!/bin/bash

python src/full_batch/run.py --conv_type $1  --undirected --dataset syn-dir --num_layers 1 --patience 500 --num_epochs 200 --num_runs 1 --lr 0.001
python src/full_batch/run.py --conv_type dir-$1 --alpha 0 --dataset syn-dir --num_layers 1 --patience 500 --num_epochs 200 --num_runs 1 --lr 0.001
python src/full_batch/run.py --conv_type dir-$1 --alpha 1  --dataset syn-dir --num_layers 1 --patience 500 --num_epochs 200 --num_runs 1 --lr 0.001
python src/full_batch/run.py --conv_type dir-$1 --alpha 0.5  --dataset syn-dir --num_layers 1 --patience 500 --num_epochs 200 --num_runs 1 --lr 0.001