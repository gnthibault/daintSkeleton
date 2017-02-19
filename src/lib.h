
//STL
#include <iostream>
#include <cstdint>
#include <numeric>
#include <iomanip>

//Boost
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>

//Local
#ifdef USE_CUDA
  #include "Cuda/lib.cu.h"
#endif //USE_CUDA

/** \struct Integrator
 * \brief operator to be mapped over a range before composing with
 * the range accumulator
 *
 * \author Thibault Notargiacomo
 */
template<typename T, typename F>
struct Integrator{
  Integrator(T step, T lowerBound, F func): m_step(step),
    m_lowerBound(lowerBound), m_func(func) {};
  T operator()(uint64_t i) const {
    return m_func(m_lowerBound+(i+0.5)*m_step);
  }
  const T m_step;
  const T m_lowerBound;
  const F m_func;
};

/** \class NumericalMidPointIntegrator1D
 * \brief Performs numerical integration on a 1D domain
 *
 * \author Thibault Notargiacomo
 */
template<typename T>
class NumericalMidPointIntegrator1D {
public:
  /// Constructor that force domain bounds definition
  NumericalMidPointIntegrator1D(T lowerBound, T upperBound, uint64_t nbSteps):
    m_lowerBound(lowerBound), m_upperBound(upperBound),
    m_nbSteps(nbSteps) {};

  /// Destructor defaulted on purpose
  virtual ~NumericalMidPointIntegrator1D()=default;

  /**
   * Function that will integrate a 1D scalar function over a 1D domain whose
   * bounds are defined by 2 fields in the object
   *
   * \param[in] f the 1D scalar function to be integrated
   *
   * \return The value of the numerical integral
   */
  template<typename F>
  T Integrate(F f) {
    T pi, sum = 0.0, step = (m_upperBound-m_lowerBound)/m_nbSteps;
    Integrator<T,F> op(step,m_lowerBound,f);
    #ifdef USE_CUDA

    #else //USE_CUDA
    sum = std::accumulate(
      boost::make_transform_iterator(
        boost::make_counting_iterator<uint64_t>(0), op),
      boost::make_transform_iterator(
        boost::make_counting_iterator<uint64_t>(m_nbSteps), op),
      0.0, std::plus<T>());
    #endif //USE_CUDA
    pi = sum*step;
    std::cout << "Pi has value "<<std::setprecision(10)<<pi<< std::endl;
    return sum;
  }

private:
  /// Lower bound for numerical integration
  const T m_lowerBound;

  /// Upper bound for numerical integration
  const T m_upperBound;

  /// Number of node used to defined the discrete grid
  const uint64_t m_nbSteps;
};


