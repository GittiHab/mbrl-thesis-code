#!/bin/bash

NAMES=("traj_all"  "traj_1000"  "traj_1000-5000")
RANGES=(""  "-r 0-1000 "  "-r 1000-5000 ")

for d in ${1}/*_newparams/; do
  for r in `seq 0 1 2`; do
    echo $d$r

    seed=10
      if [[ "$d" == *"final18"* ]]; then
        seed=18
      fi

    for i in "${!NAMES[@]}"; do
      python visualize.py -e MiniGrid-FourRooms-v0 -s ${2:-$seed} -ts -ta -n ${NAMES[i]} ${RANGES[i]}-p $d$r/replay_buffer -f pdf
    done
#    python plot_values.py -e MiniGrid-FourRooms-v0 -s ${2:-10} -b -p $d$r/model
  done
done