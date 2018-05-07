#!/usr/bin/env bash

ROOT=../   # set to your install PREFIX

NPROCS=10  # number of processes/threads
K=10       # Polynomial degree
# (this is a low value in order to keep the assembly time for the collision tensor tolerable)

printf "Set PYTHONPATH\n"
export PYTHONPATH=${ROOT}/lib/python3.6/site-packages:${PYTHONPATH}

export OMP_NUM_THREADS=${NPROCS}

printf "Compute collision tensor. This might take some time... \n Log file: tensor.out\n"
printf "Use polynomial degree K=${K}"
compute_tensor -r61 -a61 -i101 -K ${K} > tensor.out

printf "Compute initial distribution. Output: coefficients.h5"
../bin/mls --export-dofs > /dev/null && ./init_bf-step.py dof.desc

# use 1 thread per MPI process
printf "\n\n\nRun mpi computation. This might take some time... \n Log file: out\n\n"
export OMP_NUM_THREADS=1
mpirun -np "${NPROCS}" ${ROOT}/bin/mls -icoefficients.h5 &> out

# add root xml tag to solution_.xdmf
./xdmf-add-header

printf "\nComputation finished. Open `solution_.xdmf` in paraview.\n
