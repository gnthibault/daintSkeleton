#include <lib.h>

#ifdef GPU_CUDA
  #include "Cuda/lib.cu.h"
#endif //GPU_CUDA

#include <iostream>

void LibObj::libCall() {
#ifdef GPU_CUDA
  libCallCuda();
#else //GPU_CUDA
  std::cout << "This is libCall from LibObj" << std::endl;
#endif //GPU_CUDA
}
