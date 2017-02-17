#!/bin/bash

module swap PrgEnv-cray/6.0.3 PrgEnv-gnu
module load cudatoolkit
module unload daint-mc
module load daint-gpu
module load CMake/3.6.2

