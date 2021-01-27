#!/bin/bash
rm slurm-*
for i in 0 20 200 # OFC switches
# for i in 1. 5. 10. 15. 20. 25. 30. 35. 40. 45. 50. 55. 60. #MD ampflication
# for i in  1. 1.2 1.3 1.5 #0.2 0.4 0.5 0.7 0.9 G
do
  for j in {0..9} # MD LR
    do
    for k in 1. #0.6 1.  # MD active
    do
     sbatch ./slurm/run_tactile.sh $1 $i $j $k
    done
    done
done
