# daintSkeleton

## Purpose of the project
Serves as a project skeleton

## How to build for Piz Daint ?
cd daintSkeleton  
mkdir build; cd build  
mkdir $SCRATCH/MyProject  
source ../scripts/initBuild.sh  
CXX=CC CC=cc cmake -DBINDIR=$SCRATCH/MyProject -DUSE_NVCTX -DCMAKE_BUILD_TYPE=Release ..  
make -j8 install  

## How to test
In the build directory, do:  
make test  

## How to run on Piz daint
in the directory  $SCRATCH/MyProject, do  
sbatch ./launchApp1.sh
