// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef ADOLC_ADVALUE_HH
#define ADOLC_ADVALUE_HH

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if USE_BOOST_POOL
#include <boost/pool/pool_alloc.hpp>
#endif

namespace adolc {

// Utility class providing a consecutive range
// of integral values.
template <class T> class range {
  T end_;

  struct iterator {
    T pos;
    T operator*() const { return pos; }
    T &operator++() { return ++pos; }
    bool operator!=(const iterator &other) const { return pos != other.pos; }
  };

public:
  range(const T &size) : end_(size) {}
  iterator begin() const { return iterator{0}; }
  iterator end() const { return iterator{end_}; }
};

/**
 * \brief Special value to indicate a dynamically selected dimension
 *
 * This value can be passed as dimension template parameter to `ADValue`
 * to indicate that the dimension should be selected dynamically.
 */
inline constexpr std::size_t dynamicDim =
    std::numeric_limits<std::size_t>::max();

template <class V, std::size_t m, std::size_t dim = dynamicDim> class ADValue;

/**
 * \brief Traits class for checking if F is an ADValue
 * \ingroup AD
 */
template <class F> struct IsADValue;

template <class F> struct IsADValue : public std::false_type {};

template <class V, std::size_t m, std::size_t n>
struct IsADValue<ADValue<V, m, n>> : public std::true_type {};

/**
 * \brief Short cut to IsADValue<F>::value
 * \ingroup AD
 */
template <class E> constexpr bool IsADValue_t = IsADValue<E>::value;

namespace Impl {

// Helper function computing the total number of partial derivatives
static constexpr std::size_t derivativeCount(std::size_t dim,
                                             std::size_t maxOrder) {
  if (maxOrder == 1)
    return dim;
  if (maxOrder == 2)
    return dim + dim * dim;
  else
    return 0;
}

// Base class storing all derivative related data
// If a dimension is given statically, we use an
// std::array to store all partial derivatives.
template <class T, std::size_t maxOrder, std::size_t dim>
class JetDerivativeData {
public:
  static constexpr auto dimension() { return dim; }

protected:
  std::array<T, derivativeCount(dim, maxOrder)> derivatives_;
};

// If the dimension is dynamic, we either use a buffer
// managed by boost::pool<> or a std::vector as storage.
#if USE_BOOST_POOL

// This class uses a simple array-like storage of dynamic size
// with a special allocation pattern: Allocations are done using
// a boost::pool<> which provides a pool allocation strategy for
// fixed block size. To allow for dynamic sizes, this uses a static
// thread_local container storing a pool for each requested size.
// In order to avoid costly lookup for the pool, a pointer to the
// pool is stored in the object on copy and assignment.
template <class T, std::size_t maxOrder>
class JetDerivativeData<T, maxOrder, dynamicDim> {
  using Pool = boost::pool<>;

  static Pool &threadLocalPool(std::size_t size) {
    thread_local std::vector<std::unique_ptr<Pool>> pools;
    if (pools.size() < size + 1) {
      pools.resize(size + 1);
      pools[size] = std::make_unique<Pool>(size * sizeof(T));
    }
    return *(pools[size]);
  }

  T *alloc() { return reinterpret_cast<T *>(pool_->malloc()); }

  void dealloc(T *p) { pool_->free(p); }

  bool nonEmpty() const { return dimension_; }

public:
  JetDerivativeData() = default;

  JetDerivativeData(std::size_t dim, Pool &pool)
      : dimension_(dim), pool_(&pool), derivatives_(alloc()) {
    for (auto i : range(derivativeCount(dimension_, maxOrder)))
      derivatives_[i] = 0;
  }

  JetDerivativeData(std::size_t dim)
      : JetDerivativeData(dim,
                          threadLocalPool(derivativeCount(dim, maxOrder))) {}

  JetDerivativeData(const JetDerivativeData &other)
      : dimension_(other.dimension_), pool_(other.pool_) {
    if (other.nonEmpty()) {
      derivatives_ = alloc();
      for (auto i : range(derivativeCount(dimension_, maxOrder)))
        derivatives_[i] = other.derivatives_[i];
    }
  }

  JetDerivativeData(JetDerivativeData &&other)
      : dimension_(other.dimension_), pool_(other.pool_),
        derivatives_(other.derivatives_) {
    other.dimension_ = 0;
    other.pool_ = nullptr;
    other.derivatives_ = nullptr;
  }

  ~JetDerivativeData() {
    if (nonEmpty())
      dealloc(derivatives_);
  }

  JetDerivativeData &operator=(const JetDerivativeData &other) {
    if (nonEmpty()) {
      if (pool_ == other.pool_) {
        for (auto i : range(derivativeCount(dimension_, maxOrder)))
          derivatives_[i] = other.derivatives_[i];
        return *this;
      } else
        dealloc(derivatives_);
    }
    dimension_ = other.dimension_;
    pool_ = other.pool_;
    if (other.nonEmpty()) {
      derivatives_ = alloc();
      for (auto i : range(derivativeCount(dimension_, maxOrder)))
        derivatives_[i] = other.derivatives_[i];
    } else
      derivatives_ = nullptr;
    return *this;
  }

  JetDerivativeData &operator=(JetDerivativeData &&other) {
    if (nonEmpty())
      dealloc(derivatives_);
    dimension_ = other.dimension_;
    pool_ = other.pool_;
    derivatives_ = other.derivatives_;
    other.dimension_ = 0;
    other.pool_ = nullptr;
    other.derivatives_ = nullptr;
    return *this;
  }

  auto dimension() const { return dimension_; }

protected:
  std::size_t dimension_ = 0;
  Pool *pool_ = nullptr;
  T *derivatives_ = nullptr;
};

#else

template <class T, std::size_t maxOrder>
class JetDerivativeData<T, maxOrder, dynamicDim> {
public:
  JetDerivativeData() = default;

  JetDerivativeData(std::size_t dim)
      : dimension_(dim), derivatives_(derivativeCount(dim, maxOrder), 0) {}

  auto dimension() const { return dimension_; }

protected:
  std::size_t dimension_ = 0;
  std::vector<T> derivatives_;
};

#endif

template <class T, std::size_t maxOrder, std::size_t dim>
class Jet : public JetDerivativeData<T, maxOrder, dim> {
  using Base = JetDerivativeData<T, maxOrder, dim>;
  using Base::derivatives_;

public:
  using Base::Base;
  using Base::dimension;

  const auto &operator()() const { return value_; }

  auto &operator()() { return value_; }

  template <class I> const auto &operator()(I i) const {
    return derivatives_[i];
  }

  template <class I> auto &operator()(I i) { return derivatives_[i]; }

  template <class I> const auto &operator()(I i, I j) const {
    return derivatives_[dimension() * (1 + i) + j];
  }

  template <class I> auto &operator()(I i, I j) {
    return derivatives_[dimension() * (1 + i) + j];
  }

private:
  T value_;
};

} // end namespace Impl

/**
 * \brief A value for automated differentiation
 * \ingroup AD
 *
 * This class behaves like a scalar within some function
 * evaluation but simultaneously computes all derivatives
 * up to a given maximal order.
 * Currently only maxOrder<=2 is supported.
 *
 * \tparam V Type of the stored value
 * \tparam m Maximal derivative order to be computed
 * \tparam n Dimension of the domain space (defaults to dynamicDim for a
 * dynamically selected value)
 */
template <class V, std::size_t m, std::size_t dim> class ADValue {

  // Helper function for looping over supported derivative multi-indices
  template <class F> void forEachDerivativeIndex(const F &f) const {
    f();
    if constexpr (maxOrder >= 1)
      for (auto i : range(dimension()))
        f(i);
    if constexpr (maxOrder >= 2)
      for (auto i : range(dimension()))
        for (auto j : range(dimension()))
          f(i, j);
  }

  template <class F>
  static void forEachDerivativeIndex(std::size_t d, const F &f) {
    f();
    if constexpr (maxOrder >= 1)
      for (auto i : range(d))
        f(i);
    if constexpr (maxOrder >= 2)
      for (auto i : range(d))
        for (auto j : range(d))
          f(i, j);
  }

  void setDimension(ADValue &x, std::size_t d) const {
    if constexpr (dim == dynamicDim) {
      if (x.dimension() < d) {
        auto xx = x.partial();
        x.jet_ = Jet(d);
        x.partial() = xx;
      }
    }
  }

  static_assert(m <= 2, "ADValue only supports maxOrder<=2.");

public:
  static constexpr std::size_t maxOrder = m;
  using Value = V;
  using Jet = typename Impl::Jet<V, maxOrder, dim>;

  // Custom user functions. This is the basic user interface.

  /**
   * \brief Initialize as constant
   * \param value Initialize ADValue with this value
   *
   * Initialize from the given value and consider this
   * ADValue to be a constant which is invariant under
   * the function inputs.
   * This assigns all derivatives to zero.
   */
  ADValue(const Value &value) : jet_() { (*this) = value; }

  /**
   * \brief Construct from given jet
   * \param jet The jet of derivatives
   */
  ADValue(const Jet &jet) : jet_(jet) {}

  /**
   * \brief Construct from given jet
   * \param jet The jet of derivatives
   */
  ADValue(Jet &&jet) : jet_(std::move(jet)) {}

  /**
   * \brief Initialize as function input
   * \param value Initialize ADValue with this value
   * \param k Consider ADValue as k-th input variable
   *
   * Initialize from the given value and consider this
   * ADValue as k-th input variable of the function.
   * This assigns the k-th first order derivative to 1
   * and all other ones to zero.
   *
   * This constructor is only available, if the dimension
   * is statically know, i.e. if dim!=dynamicDim.
   */
  template <class T, class I,
            std::enable_if_t<(sizeof(I), dim) != dynamicDim, int> = 0>
  ADValue(const T &value, I k) : jet_() {
    partial() = value;
    if constexpr (maxOrder >= 1)
      if (dim > 0)
        partial(k) = 1;
  }

  /**
   * \brief Initialize as function input
   * \param value Initialize ADValue with this value
   * \param k Consider ADValue as k-th input variable
   * \param d The dynamic domain dimension
   *
   * Initialize from the given value and consider this
   * ADValue as k-th input variable of the function.
   * This assigns the k-th first order derivative to 1
   * and all other ones to zero.
   *
   * This constructor is only available, if the dimension
   * is dynamic know, i.e. if dim==dynamicDim.
   */
  template <class T, class I,
            std::enable_if_t<(sizeof(I), dim) == dynamicDim, int> = 0>
  ADValue(const T &value, I k, std::size_t d) : jet_(d) {
    partial() = value;
    if constexpr (maxOrder >= 1)
      if (d > 0)
        partial(k) = 1;
  }

  /**
   * \brief Assignment from raw value
   *
   * This will treat this as a constant afterwards.
   */
  template <class T, std::enable_if_t<std::is_convertible_v<T, Value>, int> = 0>
  ADValue &operator=(const T &value) {
    //  ADValue &operator=(const Value &value) {
    forEachDerivativeIndex([&](auto... i) { partial(i...) = 0; });
    partial() = value;
    return *this;
  }

  /**
   * \brief Dimension of the domain space
   */
  auto dimension() const { return jet_.dimension(); }

  /**
   * \brief Const access to stored jet
   * \returns The stored jet of derivatives
   *
   * The partial derivative can be obtained
   * from the returned object using `operator(i...)`
   * where `i... are` the indices of the derivative
   * directions.  I.e. after `ato jet = adValue.jet()`
   * one can use `jet()`, `jet(i)` and `jed(i,j)` to
   * obtain the values of the function, its first and
   * second order partial derivatives.
   */
  const auto &jet() const { return jet_; }

  /**
   * \brief Mutable access to stored jet
   * \returns The stored jet of derivatives
   *
   * The partial derivative can be obtained
   * from the returned object using `operator(i...)`
   * where `i... are` the indices of the derivative
   * directions.  I.e. after `ato jet = adValue.jet()`
   * one can use `jet()`, `jet(i)` and `jed(i,j)` to
   * obtain the values of the function, its first and
   * second order partial derivatives.
   */
  auto &jet() { return jet_; }

  /**
   * \brief Obtain `i...` partial derivative
   *
   * This is a shortcut to `jet()(i...)`.
   */
  template <class... I> const auto &partial(I... i) const { return jet_(i...); }

  /**
   * \brief Obtain `i...` partial derivative
   *
   * This is a shortcut to `jet()(i...)`.
   */
  template <class... I> auto &partial(I... i) { return jet_(i...); }

  // Explicit cast to Value

  explicit operator Value() const { return partial(); }

  explicit operator const Value &() const { return partial(); }

  // The following member functions are just standard operators
  // to make this type behave like a raw Value.

  // Use defaulted constructors and assignments

  ADValue() = default;
  ADValue(const ADValue &) = default;
  ADValue(ADValue &&) = default;
  ADValue &operator=(const ADValue &) = default;
  ADValue &operator=(ADValue &&) = default;

  // Comparison operators

  auto operator<=>(const ADValue &rhs) const {
    return partial() <=> rhs.partial();
  }

  template <class Other> auto operator<=>(const Other &rhs) const {
    return partial() <=> rhs;
  }

  bool operator==(const ADValue &rhs) const {
    return partial() == rhs.partial();
  }

  template <class Other> bool operator==(const Other &rhs) const {
    return partial() == rhs;
  }

  // Sign operators

  ADValue operator+() const { return *this; }

  ADValue operator-() const { return -1 * (*this); }

  // Addition operators

  friend ADValue operator+(const ADValue &x, const ADValue &y) {
    auto z = x;
    z += y;
    return z;
  }

  template <class Other>
  friend ADValue operator+(const ADValue &x, const Other &y) {
    auto z = x;
    z += y;
    return z;
  }

  template <class Other>
  friend ADValue operator+(const Other &x, const ADValue &y) {
    return y + x;
  }

  // Subtraction operators

  friend ADValue operator-(const ADValue &x, const ADValue &y) {
    auto z = x;
    z -= y;
    return z;
  }

  template <class Other>
  friend ADValue operator-(const ADValue &x, const Other &y) {
    auto z = x;
    z -= y;
    return z;
  }

  template <class Other>
  friend ADValue operator-(const Other &x, const ADValue &y) {
    auto z = y;
    z *= -1;
    z += x;
    return z;
  }

  // Multiplication operators

  friend ADValue operator*(const ADValue &x, const ADValue &y) {
    auto z = x;
    z *= y;
    return z;
  }

  template <class Other>
  friend ADValue operator*(const ADValue &x, const Other &y) {
    auto z = x;
    z *= y;
    return z;
  }

  template <class Other>
  friend ADValue operator*(const Other &x, const ADValue &y) {
    return y * x;
  }

  // Division operators

  friend ADValue operator/(const ADValue &x, const ADValue &y) {
    auto z = x;
    z /= y;
    return z;
  }

  template <class Other>
  friend ADValue operator/(const ADValue &x, const Other &y) {
    auto z = x;
    z /= y;
    return z;
  }

  template <class Other>
  friend ADValue operator/(const Other &x, const ADValue &y) {
    auto z = y;
    auto &Y = y.jet();
    auto &Z = z.jet();
    Z() = x / Y();
    if constexpr (maxOrder >= 1) {
      auto tmp = Y() * Y();
      for (auto i : range(y.dimension()))
        Z(i) = -x * Y(i) / tmp;
    }
    if constexpr (maxOrder >= 2) {
      auto tmp = Y() * Y() * Y();
      for (auto i : range(y.dimension()))
        for (auto j : range(y.dimension()))
          Z(i, j) = (2 * x * Y(i) * Y(j) - x * Y(i, j) * Y()) / tmp;
    }
    return z;
  }

  // Compound assignment operators

  ADValue &operator+=(const ADValue &y) {
    setDimension(*this, y.dimension());
    //    forEachDerivativeIndex([&](auto... i) {
    forEachDerivativeIndex(
        y.dimension(), [&](auto... i) { partial(i...) += y.partial(i...); });
    return *this;
  }

  template <class Other> ADValue &operator+=(const Other &y) {
    partial() += y;
    return *this;
  }

  ADValue &operator-=(const ADValue &y) {
    setDimension(*this, y.dimension());
    //    forEachDerivativeIndex([&](auto... i) {
    forEachDerivativeIndex(
        y.dimension(), [&](auto... i) { partial(i...) -= y.partial(i...); });
    return *this;
  }

  template <class Other> ADValue &operator-=(const Other &y) {
    partial() -= y;
    return *this;
  }

  ADValue &operator*=(const ADValue &y) {
    setDimension(*this, y.dimension());
    auto &X = jet();
    auto Y = [&](auto... i) -> Value {
      if (((i < y.dimension()) && ...))
        return y.partial(i...);
      else
        return Value(0);
    };
    auto &Z = jet();
    if constexpr (maxOrder >= 2)
      for (auto i : range(dimension()))
        for (auto j : range(dimension()))
          Z(i, j) = X() * Y(i, j) + Y() * X(i, j) + X(i) * Y(j) + X(j) * Y(i);
    if constexpr (maxOrder >= 1)
      for (auto i : range(dimension()))
        Z(i) = X(i) * Y() + X() * Y(i);
    Z() = X() * Y();
    return *this;
  }

  template <class Other> ADValue &operator*=(const Other &y) {
    forEachDerivativeIndex([&](auto... i) { partial(i...) *= y; });
    return *this;
  }

  ADValue &operator/=(const ADValue &y) {
    setDimension(*this, y.dimension());
    auto &X = jet();
    auto Y = [&](auto... i) -> Value {
      if (((i < y.dimension()) && ...))
        return y.partial(i...);
      else
        return Value(0);
    };
    auto &Z = jet();
    if constexpr (maxOrder >= 2) {
      auto tmp = Y() * Y() * Y();
      for (auto i : range(dimension()))
        for (auto j : range(dimension()))
          Z(i, j) =
              (X(i, j) * Y() * Y() - X() * Y(i, j) * Y() +
               2 * X() * Y(i) * Y(j) - (X(i) * Y(j) + X(j) * Y(i)) * Y()) /
              tmp;
    }
    if constexpr (maxOrder >= 1) {
      auto tmp = Y() * Y();
      for (auto i : range(dimension()))
        Z(i) = (X(i) * Y() - X() * Y(i)) / tmp;
    }
    Z() = X() / Y();
    return *this;
  }

  template <class Other> ADValue &operator/=(const Other &y) {
    forEachDerivativeIndex([&](auto... i) { partial(i...) /= y; });
    return *this;
  }

private:
  Jet jet_;
};

// Helper function for implementing unary scalar functions
template <class Value, class Derivatives>
auto adCompose(const Value &x, const Derivatives &f) {
  auto _0 = std::integral_constant<std::size_t, 0>();
  auto _1 = std::integral_constant<std::size_t, 1>();
  auto _2 = std::integral_constant<std::size_t, 2>();
  if constexpr (IsADValue_t<Value>) {
    // (f o x)'  = f'(x)*x'
    // (f o x)'' = f''(x)*x'*x' + f'(x)*x''
    auto y = x;
    auto &X = x.jet();
    auto &Y = y.jet();
    Y() = f(_0, X());
    if constexpr (Value::maxOrder >= 1) {
      auto df_x = f(_1, X());
      for (auto i : range(y.dimension()))
        Y(i) *= df_x;
      if constexpr (Value::maxOrder >= 2) {
        auto ddf_x = f(_2, X());
        for (auto i : range(x.dimension()))
          for (auto j : range(x.dimension()))
            Y(i, j) = X(i, j) * df_x + ddf_x * X(i) * X(j);
      }
    }
    return y;
  } else
    return f(_0, x);
}

// Some scalar mathematical functions

/**
 * \brief AD-aware fabs-function
 * \ingroup AD
 */
template <class X> auto fabs(const X &x) {
  using std::fabs;
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    if constexpr (order == 0)
      return fabs(x);
    else if constexpr (order == 1)
      return x < 0 ? -1. : 1.;
    else
      return 0;
  });
}

/**
 * \brief AD-aware abs-function
 * \ingroup AD
 */
template <class X> auto abs(const X &x) {
  using std::abs;
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    if constexpr (order == 0)
      return abs(x);
    else if constexpr (order == 1)
      return x < 0 ? -1. : 1.;
    else
      return 0;
  });
}

/**
 * \brief AD-aware sin-function
 * \ingroup AD
 */
template <class X> auto sin(const X &x) {
  using std::cos;
  using std::sin;
  return adCompose(x, [](auto order, auto x) {
    if ((order % 4) == 0)
      return sin(x);
    if ((order % 4) == 1)
      return cos(x);
    if ((order % 4) == 2)
      return -sin(x);
    return -cos(x);
  });
}

/**
 * \brief AD-aware cos-function
 * \ingroup AD
 */
template <class X> auto cos(const X &x) {
  using std::cos;
  using std::sin;
  return adCompose(x, [](auto order, auto x) {
    if ((order % 4) == 0)
      return cos(x);
    if ((order % 4) == 1)
      return -sin(x);
    if ((order % 4) == 2)
      return -cos(x);
    return sin(x);
  });
}

/**
 * \brief AD-aware exp-function
 * \ingroup AD
 */
template <class X> auto exp(const X &x) {
  using std::exp;
  return adCompose(x, [](auto order, auto x) { return exp(x); });
}

/**
 * \brief AD-aware log-function
 * \ingroup AD
 */
template <class X> auto log(const X &x) {
  using std::log;
  using std::pow;
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    if constexpr (order == 0)
      return log(x);
    if constexpr (order == 1)
      return 1. / x;
    if constexpr (order == 2)
      return -1. / (x * x);
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief AD-aware sqrt-function
 * \ingroup AD
 */
template <class X> auto sqrt(const X &x) {
  using std::sqrt;
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    if constexpr (order == 0)
      return sqrt(x);
    if constexpr (order == 1)
      return 1. / (2. * sqrt(x));
    if constexpr (order == 2)
      return -1. / (4. * x * sqrt(x));
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief AD-aware pow-function with constant exponent
 * \ingroup AD
 */
template <class X, class Y, std::enable_if_t<not IsADValue_t<Y>, int> = 0>
auto pow(const X &x, const Y &y) {
  using std::pow;
  return adCompose(x, [y](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    if constexpr (order == 0)
      return pow(x, y);
    if constexpr (order == 1)
      return y * pow(x, y - 1);
    if constexpr (order == 2)
      return y * (y - 1.) * pow(x, y - 2.);
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief AD-aware pow-function with constant base
 * \ingroup AD
 */
template <class X, class Y, std::enable_if_t<not IsADValue_t<X>, int> = 0>
auto pow(const X &x, const Y &y) {
  using std::log;
  using std::pow;
  return adCompose(y, [x](auto k, auto y) {
    constexpr auto order = decltype(k)::value;
    if constexpr (order == 0)
      return pow(x, y);
    if constexpr (order == 1)
      return log(x) * pow(x, y);
    if constexpr (order == 2)
      return log(x) * log(x) * pow(x, y);
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief AD-aware pow-function variable base and exponent
 * \ingroup AD
 */
template <class V, std::enable_if_t<IsADValue_t<V>, int> = 0>
auto pow(const V &x, const V &y) {
  using std::log;
  using std::pow;
  auto z = x;
  auto &X = x.jet();
  auto &Y = y.jet();
  auto &Z = z.jet();
  // Here we use:
  // z  = pow(x, y) = exp(y*log(x))
  // z' = exp(y*log(x)) * (y'*log(x) + y/x*x') = y*x^(y-1)*x' + log(x)*z*y'
  Z() = pow(X(), Y());
  if constexpr (V::maxOrder >= 1) {
    auto pow_X_Y1 = pow(X(), Y() - 1);
    auto log_X = log(X());
    auto tmp1 = Y() * pow_X_Y1;
    auto tmp2 = Z() * log_X;
    for (auto i : range(x.dimension()))
      Z(i) = tmp1 * X(i) + tmp2 * Y(i);
    if constexpr (V::maxOrder >= 2) {
      auto tmp3 = Y() * (Y() - 1) * pow(X(), Y() - 2);
      auto tmp4 = tmp2 * log_X;
      auto tmp5 = (1 + log_X * Y()) * pow_X_Y1;
      for (auto i : range(x.dimension()))
        for (auto j : range(x.dimension()))
          Z(i, j) = tmp1 * X(i, j) + tmp2 * Y(i, j) + tmp3 * X(i) * X(j) +
                    tmp4 * Y(i) * Y(j) + tmp5 * (X(i) * Y(j) + Y(i) * X(j));
    }
  }
  return z;
}

} // end namespace adolc

#endif // ADOLC_ADVALUE_HH
