# daintSkeleton

## Purpose of the project
Serves as a project skeleton

## How to build for Piz Daint ?
cd daintSkeleton
mkdir build; cd build
mkdir $SCRATCH/MyProject 
cmake -DBINDIR=$SCRATCH/MyProject ..
make install

## How to test
In the build directory, do:
make test

## How to run on Piz daint
cd $SCRATCH/MyProject 
