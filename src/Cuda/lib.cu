#include "Cuda/lib.cu.h"

//Cuda libraries includes
#include <cuda_runtime.h>

//Local
#include "Cuda/helper.cu.h"

//Kernel definition
__global__ void printKernel() {
  printf("This is libCall from LibObj from Cuda");
}

void libCallCuda() {
  PUSH_NVTX("MyGPUPrint",0)
  printKernel<<<1,1,0,0>>>();
  checkCudaErrors(cudaDeviceSynchronize());
  POP_NVTX
}
