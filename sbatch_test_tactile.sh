#!/bin/bash
for i in 0. 1. 2. 5. 10. 15. 20. 30. 40. 50. 75. 90. 100. 200. 400. #MD ampflication
do
   for j in 6.   # PFC_G
    do
     sbatch run_test_tactile.sh $1 $i $j
    done
done
