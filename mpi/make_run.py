import numpy as np

template = """
#!/bin/bash
#SBATCH -J scaling
#SBATCH -o scaling.o%j
#SBATCH -e scaling.e%j
#SBATCH -N {n_nodes}
#SBATCH -t 08:00:00
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/thejoker-benchmarks/scripts

module load gcc openmpi2 lib/hdf5/1.10.1 intel/mkl/2019-3

conda activate hq

date

mpirun -n {pool_size} python speed_scaling.py --mpi

date
"""

for pool_size in 2 ** np.arange(2, 8+1, 1):
    size = pool_size + 1
    n_nodes = size // 28
    if size / 28 > 0:
        n_nodes += 1
    run_text = template.format(pool_size=size, n_nodes=n_nodes)

    with open('run-{0:d}.sh'.format(pool_size), 'w') as f:
        f.write(run_text)
