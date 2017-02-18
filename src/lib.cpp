#include "lib.h"

#include <iostream>

#ifdef USE_CUDA
  #include "Cuda/lib.cu.h"
#endif //USE_CUDA

void LibObj::libCall() {
#ifdef USE_CUDA
  libCallCuda();
#else //USE_CUDA
  std::cout << "This is libCall from LibObj" << std::endl;
#endif //USE_CUDA
}
