#!/bin/bash
#SBATCH -J scaling
#SBATCH -o scaling.o%j
#SBATCH -e scaling.e%j
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/thejoker-benchmarks

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3

conda activate hq

date

mpirun -n 5 python speed_scaling.py --mpi

date
