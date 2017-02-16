#include "Cuda/lib.cu.h"

//Cuda libraries includes
#include <cuda_runtime.h>

//Kernel definition
__global__ void printKernel() {
  printf("This is libCall from LibObj from Cuda");
}

void libCallCuda() {
  printKernel<<<1,1,0,0>>>();
}
