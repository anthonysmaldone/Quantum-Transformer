#!/bin/bash
#SBATCH -N 2
#SBATCH --gpus-per-task=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpu-bind=none
#SBATCH -t 00:10:00
#SBATCH -q debug
#SBATCH -A m4955
#SBATCH -C gpu
#SBATCH --image=nersc/sc_cuda_quantum:24.10
#SBATCH --module=cuda-mpich

export CUDAQ_MPI_COMM_LIB=${SCRATCH}/Quantum-Transformer/distributed_interfaces/libcudaq_distributed_interface_mpi.so

# srun -N 2 -n 8 shifter bash -l 