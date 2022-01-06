#!/bin/bash
# Starts N runs with the given parameters and a random seed.
# Usage: ./launch_experiments.sh N name [parameters you would usually pass to main.py]

N=$1
NAME=$2
shift
shift
SEEDS=$RANDOM
for i in `seq 1 1 $((N-1))`; do
  SEEDS=$RANDOM,$SEEDS
done
python main.py -m hydra.job.name=$NAME "$@" environment.seed=$SEEDS > outputs/output_$NAME.txt &
echo 'Started in background:'
echo "python main.py -m hydra.job.name=$NAME $@ environment.seed=$SEEDS > outputs/output_$NAME.txt"