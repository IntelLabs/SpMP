#!/bin/bash

# Comparisong with http://www.nbi.dk/~weifeng/papers/sptrsv_liu_europar16.pdf
# Tested on Intel(R) Xeon(R) CPU E5-2699 v4 @ 2.20GHz, 2 sockets, 11 cores per socket
for i in `cat europar16.lst`; do for j in 11 22 44; do echo "OMP_NUM_THREADS=$j"; OMP_NUM_THREADS=$j ./trsv_test /scratch/jpark103/matrices/$i.mtx; done; done | tee europar16.log
