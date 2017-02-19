//STL
#include <cstdlib> 
#include <cassert> 

//Local
#include <lib.h>

int main(int argc, char* argv[]) {
  using T = double;

  //Compute Pi
  uint64_t nbSteps = 100000000;
  auto f = [](T x){ return 4./(1.+x*x);};
  NumericalMidPointIntegrator1D<T> n(0,1,nbSteps);
  auto ret = n.Integrate(f);
  
  assert(std::abs(ret-M_PI)<1.0e-12);

  return EXIT_SUCCESS;
}
