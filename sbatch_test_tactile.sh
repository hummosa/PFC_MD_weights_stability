#!/bin/bash
for i in  50. 75. 100. 200. 400. #MD ampflication
do
   for j in 6.   # PFC_G
    do
     sbatch run_test_tactile.sh $1 $i $j
    done
done
