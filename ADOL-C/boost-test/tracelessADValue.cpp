#define BOOST_TEST_DYN_LINK

#include <limits>
#include <type_traits>

#include <boost/test/unit_test.hpp>

//**************************************************
//* Test for the traceless forward mode
//* based on adolc::ADValue
//*
//* Author: Carsten Graeser
//**************************************************

//**************************************************
//* Using ADValue requires C++20 or later
//**************************************************
#if __cplusplus >= 202002L

#include <adolc/advalue.h>

//**************************************************
//* Some utilities for testing
//**************************************************

/**
 * \brief Helper function for checking an ADValue
 *
 * \param checkDim Correct dimension of argument space
 * \param value Correct value
 */
template <class T, std::size_t order, std::size_t dim>
void check(const adolc::ADValue<T, order, dim> &adValue, std::size_t checkDim,
           T value) {
  BOOST_TEST(adValue.dimension() == checkDim);
  // Check value
  BOOST_TEST(adValue.partial() == value);
}

/**
 * \brief Helper function for checking an ADValue
 *
 * \param checkDim Correct dimension of argument space
 * \param value Correct value
 * \param d Unary callback computing correct first order partial derivatives
 */
template <class T, std::size_t order, std::size_t dim, class D_Callback>
void check(const adolc::ADValue<T, order, dim> &adValue, std::size_t checkDim,
           T value, D_Callback &&d) {
  check(adValue, checkDim, value);
  // Check 1st order derivatives
  if constexpr (order >= 1)
    for (std::size_t i = 0; i < adValue.dimension(); ++i)
      BOOST_TEST(adValue.partial(i) == d(i));
}

/**
 * \brief Helper function for checking an ADValue
 *
 * \param checkDim Correct dimension of argument space
 * \param value Correct value
 * \param d Unary callback computing correct first order partial derivatives
 * \param dd Binary callback computing correct second order partial derivatives
 */
template <class T, std::size_t order, std::size_t dim, class D_Callback,
          class DD_Callback>
void check(const adolc::ADValue<T, order, dim> &adValue, std::size_t checkDim,
           T value, D_Callback &&d, DD_Callback &&dd) {
  check(adValue, checkDim, value, d);
  // Check 2nd order derivatives
  if constexpr (order >= 2)
    for (std::size_t i = 0; i < adValue.dimension(); ++i)
      for (std::size_t j = 0; j < adValue.dimension(); ++j)
        BOOST_TEST(adValue.partial(i, j) == dd(i, j));
}

/**
 * \brief Helper function object returning zero for any input arguments
 */
constexpr auto zero = [](auto... i) { return 0; };

/**
 * \brief Create example value of static size
 *
 * The resulting ADValue's value is set to seed,
 * while the (i0,...in)-th partial derivative
 * is seed*i0*...*in.
 */
template <class T, std::size_t order, std::size_t dim>
auto exampleValue(T seed) {
  auto x = adolc::ADValue<T, order, dim>();
  x.partial() = seed;
  if constexpr (order >= 1) {
    for (std::size_t i = 0; i < dim; ++i)
      x.partial(i) = seed * i;
  }
  if constexpr (order >= 2) {
    for (std::size_t i = 0; i < dim; ++i)
      for (std::size_t j = 0; j < dim; ++j)
        x.partial(i, j) = seed * i * j;
  }
  return x;
}

/**
 * \brief Create example value of dynamic size
 *
 * The resulting ADValue's value is set to seed,
 * while the (i0,...in)-th partial derivative
 * is seed*i0*...*in.
 */
template <class T, std::size_t order>
auto exampleValue(std::size_t dim, T seed) {
  if constexpr (order == 0)
    return adolc::ADValue<T, order, adolc::dynamicDim>(seed);
  else {
    if (dim == 0)
      return adolc::ADValue<T, order, adolc::dynamicDim>(seed);
    auto x = adolc::ADValue<T, order, adolc::dynamicDim>(seed, 0, dim);
    if constexpr (order >= 1) {
      for (std::size_t i = 0; i < dim; ++i)
        x.partial(i) = seed * i;
    }
    if constexpr (order >= 2) {
      for (std::size_t i = 0; i < dim; ++i)
        for (std::size_t j = 0; j < dim; ++j)
          x.partial(i, j) = seed * i * j;
    }
    return x;
  }
}

/**
 * \brief Call check with a few combinations of base type order and dimension
 */
template <class Check> void defaultTestSuite(Check &&checkCallBack) {
  checkCallBack.template operator()<double, 0, 0>();
  checkCallBack.template operator()<double, 1, 0>();
  checkCallBack.template operator()<double, 1, 1>();
  checkCallBack.template operator()<double, 1, 2>();
  checkCallBack.template operator()<double, 1, 3>();
  checkCallBack.template operator()<double, 2, 0>();
  checkCallBack.template operator()<double, 2, 1>();
  checkCallBack.template operator()<double, 2, 2>();
  checkCallBack.template operator()<double, 2, 3>();

  checkCallBack.template operator()<float, 0, 0>();
  checkCallBack.template operator()<float, 1, 0>();
  checkCallBack.template operator()<float, 1, 1>();
  checkCallBack.template operator()<float, 1, 2>();
  checkCallBack.template operator()<float, 1, 3>();
  checkCallBack.template operator()<float, 2, 0>();
  checkCallBack.template operator()<float, 2, 1>();
  checkCallBack.template operator()<float, 2, 2>();
  checkCallBack.template operator()<float, 2, 3>();
}

//**************************************************
//* Test of individual feature of ADValue
//**************************************************

BOOST_AUTO_TEST_SUITE(traceless_advalue)

BOOST_AUTO_TEST_CASE(ADValueConstructor) {

#if USE_BOOST_POOL
  std::cout << "Testing ADValue with boost-pool support." << std::endl;
#else
  std::cout << "Testing ADValue without boost-pool support." << std::endl;
#endif

  defaultTestSuite([]<class T, std::size_t order, std::size_t dim>() {
    // Check default dimension value
    using ADValue_dynamic = adolc::ADValue<T, order, adolc::dynamicDim>;
    using ADValue_default = adolc::ADValue<T, order>;
    BOOST_TEST((std::is_same_v<ADValue_dynamic, ADValue_default>));

    // Construct as constant
    T x = 42;
    check(adolc::ADValue<T, order, dim>(x), dim, x, zero, zero);
    check(adolc::ADValue<T, order, adolc::dynamicDim>(x), 0, x, zero, zero);

    if constexpr (order >= 1) {
      // Construct as k-th input argument
      for (std::size_t k = 0; k < dim; ++k) {
        auto D = [&](auto i) { return i == k; };
        check(adolc::ADValue<T, order, dim>(x, k), dim, x, D, zero);
        check(adolc::ADValue<T, order, adolc::dynamicDim>(x, k, dim), dim, x, D,
              zero);
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(ADValueAssign) {
  defaultTestSuite([]<class T, std::size_t order, std::size_t dim>() {
    T aValue = 42;
    T bValue = 23;
    T cValue = 237;

    {
      auto a = adolc::ADValue<T, order, dim>(aValue);
      auto b = adolc::ADValue<T, order, dim>(bValue);
      check(a, dim, aValue, zero, zero);
      a = b;
      check(a, dim, bValue, zero, zero);
      a = cValue;
      check(a, dim, cValue, zero, zero);
    }
    {
      auto a = adolc::ADValue<T, order, adolc::dynamicDim>(aValue);
      auto b = adolc::ADValue<T, order, adolc::dynamicDim>(bValue);
      check(a, 0, aValue, zero, zero);
      a = b;
      check(a, 0, bValue, zero, zero);
      a = cValue;
      check(a, 0, cValue, zero, zero);
    }

    if constexpr ((dim > 0) and (order > 0)) {
      {
        auto a = adolc::ADValue<T, order, dim>(aValue, 0);
        auto b = adolc::ADValue<T, order, dim>(bValue, dim - 1);
        check(a, dim, aValue, [](auto i) { return i == 0; }, zero);
        a = b;
        check(a, dim, bValue, [](auto i) { return i == (dim - 1); }, zero);
        a = cValue;
        check(a, dim, cValue, zero, zero);
      }
      {
        auto a = adolc::ADValue<T, order, adolc::dynamicDim>(aValue, 0, 1);
        auto b =
            adolc::ADValue<T, order, adolc::dynamicDim>(bValue, dim - 1, dim);
        check(a, 1, aValue, [](auto i) { return i == 0; }, zero);
        a = b;
        check(a, dim, bValue, [](auto i) { return i == (dim - 1); }, zero);
        a = cValue;
        check(a, dim, cValue, zero, zero);
      }
    }
  });
}

BOOST_AUTO_TEST_CASE(ADValueSum) {
  defaultTestSuite([]<class T, std::size_t order, std::size_t dim>() {
    T aValue = 42;
    T bValue = 23;

    {
      auto a = exampleValue<T, order, dim>(aValue);
      auto b = exampleValue<T, order, dim>(bValue);

      auto c = a + b;
      check(
          c, dim, aValue + bValue,
          [&](auto i) { return (aValue + bValue) * i; },
          [&](auto i, auto j) { return (aValue + bValue) * i * j; });

      auto d = a;
      d += b;
      check(
          d, dim, aValue + bValue,
          [&](auto i) { return (aValue + bValue) * i; },
          [&](auto i, auto j) { return (aValue + bValue) * i * j; });
    }

    {
      auto a = exampleValue<T, order>(dim, aValue);
      auto b = exampleValue<T, order>(dim, bValue);

      auto c = a + b;
      check(
          c, dim, aValue + bValue,
          [&](auto i) { return (aValue + bValue) * i; },
          [&](auto i, auto j) { return (aValue + bValue) * i * j; });

      auto d = a;
      d += b;
      check(
          d, dim, aValue + bValue,
          [&](auto i) { return (aValue + bValue) * i; },
          [&](auto i, auto j) { return (aValue + bValue) * i * j; });
    }
  });
}

BOOST_AUTO_TEST_CASE(ADValueDiff) {
  defaultTestSuite([]<class T, std::size_t order, std::size_t dim>() {
    T aValue = 42;
    T bValue = 23;

    {
      auto a = exampleValue<T, order, dim>(aValue);
      auto b = exampleValue<T, order, dim>(bValue);

      auto c = a - b;
      check(
          c, dim, aValue - bValue,
          [&](auto i) { return (aValue - bValue) * i; },
          [&](auto i, auto j) { return (aValue - bValue) * i * j; });

      auto d = a;
      d -= b;
      check(
          d, dim, aValue - bValue,
          [&](auto i) { return (aValue - bValue) * i; },
          [&](auto i, auto j) { return (aValue - bValue) * i * j; });
    }

    {
      auto a = exampleValue<T, order>(dim, aValue);
      auto b = exampleValue<T, order>(dim, bValue);

      auto c = a - b;
      check(
          c, dim, aValue - bValue,
          [&](auto i) { return (aValue - bValue) * i; },
          [&](auto i, auto j) { return (aValue - bValue) * i * j; });

      auto d = a;
      d -= b;
      check(
          d, dim, aValue - bValue,
          [&](auto i) { return (aValue - bValue) * i; },
          [&](auto i, auto j) { return (aValue - bValue) * i * j; });
    }
  });
}

BOOST_AUTO_TEST_CASE(ADValueMult) {
  defaultTestSuite([]<class T, std::size_t order, std::size_t dim>() {
    T aValue = 42;
    T bValue = 23;

    {
      auto a = exampleValue<T, order, dim>(aValue);
      auto b = exampleValue<T, order, dim>(bValue);

      auto c = a * b;
      check(
          c, dim, aValue * bValue,
          [&](auto i) { return 2 * aValue * bValue * i; },
          [&](auto i, auto j) { return 4 * aValue * bValue * i * j; });

      auto d = a;
      d *= b;
      check(
          d, dim, aValue * bValue,
          [&](auto i) { return 2 * aValue * bValue * i; },
          [&](auto i, auto j) { return 4 * aValue * bValue * i * j; });
    }

    {
      auto a = exampleValue<T, order>(dim, aValue);
      auto b = exampleValue<T, order>(dim, bValue);

      auto c = a * b;
      check(
          c, dim, aValue * bValue,
          [&](auto i) { return 2 * aValue * bValue * i; },
          [&](auto i, auto j) { return 4 * aValue * bValue * i * j; });

      auto d = a;
      d *= b;
      check(
          d, dim, aValue * bValue,
          [&](auto i) { return 2 * aValue * bValue * i; },
          [&](auto i, auto j) { return 4 * aValue * bValue * i * j; });
    }
  });
}

//**************************************************
//* ToDo: Add checks for the following features
//* - Division of ADValue and ADValue
//* - Arithmetic operations of ADValue and raw value
//* - Nonlinear functions
//**************************************************

BOOST_AUTO_TEST_SUITE_END()

#else //__cplusplus >= 202002L

BOOST_AUTO_TEST_SUITE(traceless_advalue)
BOOST_AUTO_TEST_CASE(ADValueNotTested) {
  std::cout << "ADOL-C Warning: ADValue is not tested since test is not "
               "compiled with C++20 support."
            << std::endl;
}
BOOST_AUTO_TEST_SUITE_END()

#endif //__cplusplus >= 202002L
