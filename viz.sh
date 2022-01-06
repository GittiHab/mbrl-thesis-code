#!/bin/bash
regex=".*explore_([0-9]+)_steps\.zip"
#for r in `seq 0 1 2`; do
#  for i in `seq 2000 1000 10000`; do
#    python plot_values.py -e MiniGrid-FourRooms-v0 -s ${2:-10} -b -r -n rewards_$i -p ${1}/$r/checkpoints/model_explore_$i\_steps.zip
#    python plot_values.py -e MiniGrid-FourRooms-v0 -s ${2:-10} -b -n values_$i -p ${1}/$r/checkpoints/model_explore_$i\_steps.zip
#  done
#  for i in ${1}/$r/checkpoints/model_explore_*.zip; do
#  for i in ${1}/model_explore_*.zip; do
d=${1}
#for d in ${1}/*_newparams/; do
  for r in `seq 0 1 2`; do
#    for i in $d$r/checkpoints/model_explore_*.zip; do
      i=$d/$r/model.zip
      seed=10
      if [[ "$i" == *"final18"* ]]; then
        seed=18
      fi

#      if [[ $i =~ $regex ]]
#      then
        timestep="${BASH_REMATCH[1]}"
#        python plot_values.py -e MiniGrid-FourRooms-v0 -s $seed -b -r -n rewards_$timestep -p $i -f pdf
#        python plot_values.py -e MiniGrid-FourRooms-v0 -s $seed -b -n values_$timestep -p $i
        python plot_values.py -e MiniGrid-FourRooms-v0 -s $seed -b -n values_$timestep -p $i -f pdf
#      else
#        echo "Regex didn't match!"
#      fi
#    done
  done
#done