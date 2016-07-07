#!/bin/bash

for i in `cat europar16.lst`; do for j in 11 22 44; do echo "OMP_NUM_THREADS=$j"; OMP_NUM_THREADS=$j ./trsv_test /scratch/jpark103/matrices/$i.mtx; done; done | tee europar16.log
