
/** \class LibObj
 * \brief Dummy class used to illustrate a c++ library project
 *
 * \author John Doe
 */

class LibObj {
public:
  /// Constructor defaulted on purpose
  LibObj()=default;
  /// Destructor defaulted on purpose
  virtual ~LibObj()=default;

  /**
   *
   * Dummy function that should simply print out if it was executed
   * on a regular CPU, or on GPU
   *
   * \param[in] arg1 that's where arg1 description should go if there was one
   *
   * \param[out] arg2 that's where arg1 description should go if there was one
   *
   * \return Actually nothing
   */
  void libCall();
};
