#!/bin/bash
for i in 1.2  1.4 1.8 3.1 
do
   for j in 2.2 2.4 3.4 
    do
     sbatch run_tactile_GPU.sh $i $j
    done
done
