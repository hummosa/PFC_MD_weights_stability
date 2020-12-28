#!/bin/bash
rm slurm-*
for i in  30. #MD ampflication
# for i in 1. 5. 10. 15. 20. 25. 30. 35. 40. 45. 50. 55. 60. #MD ampflication
# for i in  1. 1.2 1.3 1.5 #0.2 0.4 0.5 0.7 0.9 G
do
  for j in 0 1 #2 3 4 5 6 7 8 9 # MD LR
    do
    for k in 1. 2. #0.1 0.6 1.  # MD bias factor
    do
     sbatch ./slurm/run_tactile.sh $1 $i $j $k
    done
    done
done
