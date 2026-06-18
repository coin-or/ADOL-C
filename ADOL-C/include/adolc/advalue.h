// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef ADOLC_ADVALUE_HH
#define ADOLC_ADVALUE_HH

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <adolc/internal/adolc_settings.h>

#if USE_BOOST_POOL
#include <boost/pool/pool_alloc.hpp>
#endif

namespace adolc {

// Utility class providing a consecutive range
// of integral values.
template <class T> class range {
  T begin_;
  T end_;

  struct iterator {
    T pos;
    T operator*() const { return pos; }
    T &operator++() { return ++pos; }
    bool operator!=(const iterator &other) const { return pos != other.pos; }
  };

public:
  range(const T &begin, const T &end) : begin_(begin), end_(end) {}
  range(const T &size) : range(T(0), size) {}
  iterator begin() const { return iterator{begin_}; }
  iterator end() const { return iterator{end_}; }
};

/**
 * \brief Special value to indicate a dynamically selected dimension
 * \ingroup TapeLess
 *
 * This value can be passed as dimension template parameter to `ADValue`
 * to indicate that the dimension should be selected dynamically.
 */
inline constexpr std::size_t dynamicDim =
    std::numeric_limits<std::size_t>::max();

template <class V, std::size_t m, std::size_t dim = dynamicDim> class ADValue;

/**
 * \brief Traits class for checking if F is an `ADValue`
 * \ingroup TapeLess
 */
template <class F> struct IsADValue;

template <class F> struct IsADValue : public std::false_type {};

template <class V, std::size_t m, std::size_t n>
struct IsADValue<ADValue<V, m, n>> : public std::true_type {};

/**
 * \brief Short cut to `IsADValue<F>::value`
 * \ingroup TapeLess
 */
template <class E> constexpr bool IsADValue_t = IsADValue<E>::value;

namespace Impl {

// Helper function computing the total number of partial derivatives
static constexpr std::size_t fullDerivativeCount(std::size_t dim,
                                                 std::size_t maxOrder) {
  if (maxOrder == 1)
    return dim;
  if (maxOrder == 2)
    return dim + dim * dim;
  else
    return 0;
}

// Helper function computing the reduced number of partial derivatives
// exploiting symmetry
static std::size_t reducedDerivativeCount(std::size_t dim,
                                          std::size_t maxOrder) {
  if (maxOrder == 1)
    return dim;
  if (maxOrder == 2)
    return dim + (dim + 1) * dim / 2;
  else
    return 0;
}

// Base class storing derivatives up to given maxOrder.
// If the dimension is given statically, we use an
// std::array to store all partial derivatives.
// This implementation stores the full Hessian,
// including duplicates of the mixed derivatives.
//
// For statically known size the compiler generates
// faster code for this, probably by using auto-vectorization
// with SIMD instructions.
template <class T, std::size_t maxOrder, std::size_t dim>
class DerivativeStorage
    : public std::array<T, fullDerivativeCount(dim, maxOrder)> {
  using Base = std::array<T, fullDerivativeCount(dim, maxOrder)>;

public:
  using Base::operator[];
  using Base::size;

  // Export the dimension.
  static constexpr std::size_t dimension() { return dim; }

  void updateDimension(const DerivativeStorage &other) {}

  // Given an index i, this exports the number n, such that
  // all derivatives (i,j) with j<n are stored.
  static constexpr std::size_t storedDerivatives(std::size_t i) { return dim; }

  // The index method translates a n-tuple of indices representing
  // an n-th order derivative into the internal flat storage index.
  static constexpr std::size_t storedDerivativeIndex(std::size_t i) {
    return i;
  }

  static constexpr std::size_t storedDerivativeIndex(std::size_t i,
                                                     std::size_t j) {
    return dim * (1 + i) + j;
  }

  template <class... I> static constexpr std::size_t derivativeIndex(I... i) {
    return storedDerivativeIndex(i...);
  }
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
template <class T> class PoolVector {
  class Pool : public boost::pool<> {
  public:
    Pool(std::size_t size) : boost::pool<>(size * sizeof(T)), size_(size) {}

    std::size_t size() const { return size_; }

    T *alloc() { return reinterpret_cast<T *>(this->malloc()); }

    void dealloc(T *p) { this->free(p); }

  private:
    std::size_t size_;
  };

  static Pool &threadLocalPool(std::size_t size) {
    thread_local std::vector<std::unique_ptr<Pool>> pools;
    if (pools.size() < size + 1) {
      pools.resize(size + 1);
      pools[size] = std::make_unique<Pool>(size);
    }
    return *(pools[size]);
  }

  bool nonEmpty() const { return pool_; }

public:
  PoolVector() = default;

  PoolVector(std::size_t size, const T &defaultValue)
      : pool_(&threadLocalPool(size)) {
    data_ = pool_->alloc();
    for (auto i : range(size))
      data_[i] = defaultValue;
  }

  PoolVector(Pool *pool) : pool_(pool) {
    if (pool_)
      data_ = pool_->alloc();
  }

  PoolVector(Pool *pool, const T &defaultValue) : pool_(pool) {
    if (pool_) {
      data_ = pool_->alloc();
      for (auto i : range(size()))
        data_[i] = defaultValue;
    }
  }

  PoolVector(const PoolVector &other) : pool_(other.pool_) {
    if (other.nonEmpty()) {
      data_ = pool_->alloc();
      for (auto i : range(size()))
        data_[i] = other.data_[i];
    }
  }

  PoolVector(PoolVector &&other) : pool_(other.pool_), data_(other.data_) {
    other.pool_ = nullptr;
    other.data_ = nullptr;
  }

  ~PoolVector() {
    if (nonEmpty())
      pool_->dealloc(data_);
  }

  PoolVector &operator=(const PoolVector &other) {
    if (nonEmpty()) {
      if (pool_ == other.pool_) {
        for (auto i : range(size()))
          data_[i] = other.data_[i];
        return *this;
      } else
        pool_->dealloc(data_);
    }
    pool_ = other.pool_;
    if (other.nonEmpty()) {
      data_ = pool_->alloc();
      for (auto i : range(size()))
        data_[i] = other.data_[i];
    } else
      data_ = nullptr;
    return *this;
  }

  PoolVector &operator=(PoolVector &&other) {
    if (nonEmpty())
      pool_->dealloc(data_);
    pool_ = other.pool_;
    data_ = other.data_;
    other.pool_ = nullptr;
    other.data_ = nullptr;
    return *this;
  }

  void resize(std::size_t newSize, const T &defaultValue) {
    auto *newPool = &threadLocalPool(newSize);
    auto *newDerivatives = newPool->alloc();
    if (nonEmpty()) {
      for (auto i : range(size()))
        newDerivatives[i] = data_[i];
      pool_->dealloc(data_);
    }
    for (auto i : range(size(), newSize))
      newDerivatives[i] = defaultValue;
    pool_ = newPool;
    data_ = newDerivatives;
  }

  T &operator[](std::size_t k) { return data_[k]; }

  const T &operator[](auto k) const { return data_[k]; }

  std::size_t size() const {
    if (pool_)
      return pool_->size();
    else
      return 0;
  }

  Pool *pool() const { return pool_; }

protected:
  Pool *pool_ = nullptr;
  T *data_ = nullptr;
};

template <class T> using DynamicArray = PoolVector<T>;

template <class T, std::size_t m>
auto derivativeStorageWithSameDimension(
    const DerivativeStorage<T, m, dynamicDim> &other) {
  return DerivativeStorage<T, m, dynamicDim>(other.dimension(), other.pool());
}

#else

template <class T> using DynamicArray = std::vector<T>;

template <class T, std::size_t m>
auto derivativeStorageWithSameDimension(
    const DerivativeStorage<T, m, dynamicDim> &other) {
  return DerivativeStorage<T, m, dynamicDim>(other.dimension());
}

#endif

// If the dimension is given dynamically, we use an
// std::vector or related boost::pool<> based container
// to store the partial derivatives.
// This implementation only stores the lower triangle
// of the Hessian, avoiding duplicates of the mixed derivatives.
template <class T, std::size_t maxOrder>
class DerivativeStorage<T, maxOrder, dynamicDim> : public DynamicArray<T> {
  using Base = DynamicArray<T>;
  using Base::resize;

public:
  using Base::operator[];
  using Base::size;

  DerivativeStorage() = default;

  DerivativeStorage(const DerivativeStorage &other) = default;

  DerivativeStorage(DerivativeStorage &&other)
      : Base(std::move(other)), dimension_(other.dimension_) {
    other.dimension_ = 0;
  }

  explicit DerivativeStorage(std::size_t dim)
      : Base(reducedDerivativeCount(dim, maxOrder), 0), dimension_(dim) {}

  template <class... Args, std::enable_if_t<(sizeof...(Args) > 0), int> = 0>
  DerivativeStorage(std::size_t dim, Args &&...args)
      : Base(std::forward<Args>(args)...), dimension_(dim) {}

  DerivativeStorage &operator=(const DerivativeStorage &other) = default;

  DerivativeStorage &operator=(DerivativeStorage &&other) {
    Base::operator=(std::move(other));
    dimension_ = other.dimension_;
    other.dimension_ = 0;
    return *this;
  }

  std::size_t dimension() const { return dimension_; }

  void updateDimension(const DerivativeStorage &other) {
    if (dimension() < other.dimension()) {
      auto copy = DerivativeStorage(other.dimension());
      if constexpr (maxOrder >= 1)
        for (auto i : range(dimension()))
          copy[copy.storedDerivativeIndex(i)] =
              (*this)[storedDerivativeIndex(i)];
      if constexpr (maxOrder >= 2)
        for (auto i : range(dimension()))
          for (auto j : range(storedDerivatives(i)))
            copy[copy.storedDerivativeIndex(i, j)] =
                (*this)[storedDerivativeIndex(i, j)];
      *this = copy;
    }
  }

  // Given an index i, this exports the number n, such that
  // all derivatives (i,j) with j<n are stored.
  auto storedDerivatives(std::size_t i) const { return i + 1; }

  // The index method translates a n-tuple of indices representing
  // an n-th order derivative into the internal flat storage index.
  std::size_t storedDerivativeIndex(std::size_t i) const { return i; }

  std::size_t storedDerivativeIndex(std::size_t i, std::size_t j) const {
    return dimension_ + i * (i + 1) / 2 + j;
  }

  std::size_t derivativeIndex(std::size_t i) const {
    return storedDerivativeIndex(i);
  }

  std::size_t derivativeIndex(std::size_t i, std::size_t j) const {
    if (i >= j)
      return storedDerivativeIndex(i, j);
    else
      return storedDerivativeIndex(j, i);
  }

protected:
  std::size_t dimension_ = 0;
};

// Generate a convenience function for accessing the
// derivatives stored in an `ADValue` from indices for
// the deriavative directions. In contranst to adValue.partial(i...)
// this only supports to pass indices i... that are
// actually stored.
template <class ADV> auto storedDerivativeAccess(ADV &adValue) {
  return [&](auto... i) -> decltype(auto) {
    if constexpr (sizeof...(i) == 0)
      return adValue.value();
    else
      return adValue.derivativeStorage()[adValue.derivativeStorage()
                                             .storedDerivativeIndex(i...)];
  };
}

template <class ADV> auto paddedDerivativeAccess(ADV &adValue) {
  using Value = typename ADV::Value;
  return [&](auto... i) -> Value {
    if (((i < adValue.derivativeStorage().dimension()) && ...))
      return storedDerivativeAccess(adValue)(i...);
    else
      return Value(0);
  };
}

template <class V, std::size_t m, std::size_t dim>
auto createWithMaxDimension(const ADValue<V, m, dim> &x,
                            const ADValue<V, m, dim> &y) {
  if constexpr (dim == dynamicDim) {
    if (x.dimension() >= y.dimension())
      return ADValue<V, m, dim>(
          0, derivativeStorageWithSameDimension(x.derivativeStorage()));
    else
      return ADValue<V, m, dim>(
          0, derivativeStorageWithSameDimension(y.derivativeStorage()));
  } else
    return ADValue<V, m, dim>();
}

} // end namespace Impl

// Forward declaration
template <class X> X inv(const X &x);

/**
 * \brief A value class for traceless automated differentiation
 * \ingroup TapeLess
 *
 * \tparam V Type of the stored value
 * \tparam m Maximal derivative order to be computed
 * \tparam n Dimension of the domain space (defaults to `dynamicDim` for a
dynamically selected value)
 *
 * This class behaves like a scalar within some function
 * evaluation but simultaneously computes all derivatives
 * up to a given maximal order.
 * Currently only `m<=2` is supported.
 *
 * The following provides a usage example, where this class is used
 * to evaluate second order derivatives of a trivariate function.
 * \code{.cpp}
// Use double as underlying type
using T = double;

// Raw input vector for function evaluation
auto x_raw = std::array<T, 3>{ 23., 42., 237. };

// Function to evaluate
auto f = [](auto x) {
  return log(1. + x[0]*x[0]) / (2. + cos(x[1])*sin(x[2]));
};

// Typedef to AD-aware version of T. This will track up to
// second order derivatives with respect to three directions.
// If the number of directions is omitted, it will be determined
// dynamically, which is in general significantly slower.
using T_AD = ADValue<double, 2, 3>;

// Translate x to AD-aware version
auto x = std::array<T_AD, 3>();
for(std::size_t i=0; i<3; ++i)
  x[i] = T_AD(x_raw[i], i);

// Evaluate the function
auto y = f(x);

// Print plain function value. Alternatively to y.value()
// we could use y.value() or the cast T(y)
std::cout << y.value() << std::endl;

// Print all first order partial derivatives
for(std::size_t i=0; i<3; ++i)
  std::cout << y.partial(i) << std::endl;

// Print all second order partial derivatives
for(std::size_t i=0; i<3; ++i)
  for(std::size_t j=0; j<3; ++j)
    std::cout << y.partial(i, j) << std::endl;
 * \endcode
 */
template <class V, std::size_t m, std::size_t dim> class ADValue {

  static_assert(m <= 2, "ADValue only supports maxOrder<=2.");

public:
  //! Maximal order of derivatives that will be tracked
  static constexpr std::size_t maxOrder = m;

  //! Underlying value type
  using Value = V;

  //! Interbal type for flat storage of partial derivatives
  using DerivativeStorage = typename Impl::DerivativeStorage<V, maxOrder, dim>;

  // Constructors

  /**
   * \brief Defaulted default constructor
   *
   * This will default initialize the stored value
   * and partial derivatives. If `dim==dynamicDim`,
   * then dynamic dimension value will be initialized to 0.
   */
  ADValue() = default;

  //! Defaulted copy constructor
  ADValue(const ADValue &) = default;

  //! Defaulted move constructor
  ADValue(ADValue &&) = default;

  /**
   * \brief Initialize as constant
   * \param value Initialize `ADValue` with this value
   *
   * Initialize from the given value and consider this
   * `ADValue` to be a constant which is invariant under
   * the function inputs.
   * This assigns all derivatives to zero.
   */
  template <class T, std::enable_if_t<std::is_convertible_v<T, Value>, int> = 0>
  ADValue(const T &value) : derivatives_() {
    (*this) = value;
  }

  /**
   * \brief Initialize as function input
   * \param value Initialize `ADValue` with this value
   * \param k Consider `ADValue` as `k`-th input variable
   *
   * Initialize from the given value and consider this
   * `ADValue` as `k`-th input variable of the function.
   * This assigns the k-th first order derivative to 1
   * and all other ones to zero.
   *
   * This constructor is only available, if the dimension
   * is statically know, i.e. if `dim!=dynamicDim` and if `maxOrder>0`.
   */
  template <class T,
            std::enable_if_t<
                ((sizeof(T), dim) != dynamicDim) and (maxOrder > 0), int> = 0>
  ADValue(const T &value, std::size_t k) : derivatives_() {
    value_ = value;
    if constexpr (maxOrder >= 1)
      derivatives_[derivatives_.storedDerivativeIndex(k)] = 1;
  }

  /**
   * \brief Initialize as function input
   * \param value Initialize `ADValue` with this value
   * \param k Consider `ADValue` as `k`-th input variable
   *
   * Initialize from the given value and consider this
   * `ADValue` as `k`-th input variable of the function.
   * This assigns the `k`-th first order derivative to 1
   * and all other ones to zero.
   *
   * This constructor is only available, if the dimension
   * is dynamic, i.e. if `dim==dynamicDim` and if `maxOrder>0`.
   * It will initialize the dynamic dimension value by `k+1`.
   */
  template <class T,
            std::enable_if_t<
                ((sizeof(T), dim) == dynamicDim) and (maxOrder > 0), int> = 0>
  ADValue(const T &value, std::size_t k) : ADValue(value, k, k + 1) {}

  /**
   * \brief Initialize as function input
   * \param value Initialize `ADValue` with this value
   * \param k Consider `ADValue` as `k`-th input variable
   * \param d The dynamic domain dimension
   *
   * Initialize from the given value and consider this
   * `ADValue` as `k`-th input variable of the function.
   * This assigns the `k`-th first order derivative to 1
   * and all other ones to zero.
   *
   * This constructor is only available, if the dimension
   * is dynamic know, i.e. if `dim==dynamicDim` and if `maxOrder>0`.
   */
  template <class T,
            std::enable_if_t<
                ((sizeof(T), dim) == dynamicDim) and (maxOrder > 0), int> = 0>
  ADValue(const T &value, std::size_t k, std::size_t d) : derivatives_(d) {
    value_ = value;
    if constexpr (maxOrder >= 1)
      derivatives_[derivatives_.storedDerivativeIndex(k)] = 1;
  }

  /**
   * \brief Initialize from raw value and derivatives data
   * \param value Initializer for value
   * \param derivatives Initializer for partial derivative storage
   */
  template <class T>
  ADValue(const T &value, const DerivativeStorage &derivatives)
      : value_(value), derivatives_(derivatives) {}

  /**
   * \brief Initialize from raw value and derivatives data
   * \param value Initializer for value
   * \param derivatives Initializer for partial derivative storage
   */
  template <class T>
  ADValue(const T &value, DerivativeStorage &&derivatives)
      : value_(value), derivatives_(std::move(derivatives)) {}

  /**
   * \brief Assignment from raw value
   *
   * This will treat this as a constant afterwards.
   */
  template <class T, std::enable_if_t<std::is_convertible_v<T, Value>, int> = 0>
  ADValue &operator=(const T &value) {
    value_ = value;
    for (auto k : range(derivatives_.size()))
      derivatives_[k] = 0;
    return *this;
  }

  // Assignment operators

  //! Defaulted copy assignment
  ADValue &operator=(const ADValue &) = default;

  //! Defaulted move assignment
  ADValue &operator=(ADValue &&) = default;

  // Comparison operators

  //! Three way comparison based on the stored value
  auto operator<=>(const ADValue &rhs) const { return value_ <=> rhs.value_; }

  //! Three way comparison based on the stored value
  template <class Other> auto operator<=>(const Other &rhs) const {
    return value_ <=> rhs;
  }

  //! Equality comparison based on the stored value
  bool operator==(const ADValue &rhs) const { return value_ == rhs.value_; }

  //! Equality comparison based on the stored value
  template <class Other> bool operator==(const Other &rhs) const {
    return value_ == rhs;
  }

  // Custom user interface

  //! Dimension of the domain space
  auto dimension() const { return derivatives_.dimension(); }

  //! Const access to value
  const Value &value() const { return value_; }

  //! Mutable access to value
  Value &value() { return value_; }

  /**
   * \brief Mutable access to stored derivatives
   * \returns Container with stored derivatives
   *
   * The partial derivatives are stored in an implementation
   * defined flat container.
   * From the returned storage object `s = adValue.derivativeStorage()`
   * the derivative in directions `i, j, ...` can be obtained
   * using `s[s.derivativeIndex(i, j, ...)]`. Notice that,
   * depending on the internal storage, this may return a
   * reference to the same storage location for potentially
   * duplicate derivatives, i.e. `s[s.derivativeIndex(1,0)]`
   * may internally flip the indices and return the same
   * location as `s[s.derivativeIndex(0,1)]`. In contrast
   * `s[s.storedDerivativeIndex(i, j, ...)]` will not do
   * any index flipping, but only supports to pass the
   * indices of actually stored derivatives.
   */
  auto &derivativeStorage() { return derivatives_; }

  /**
   * \brief Const access to stored derivatives
   * \returns Container with stored derivatives
   *
   * This has the same semantics as the non-const
   * method with the same name but only provides
   * const access to the stored derivatives.
   */
  const auto &derivativeStorage() const { return derivatives_; }

  /**
   * \brief Obtain `i...` partial derivative
   *
   * `adValue.partial(i,j, ...)` returns the partial
   * derivative in directions `i,j,...`. Passing no
   * arguments returns the value.
   */
  template <class... I> const auto &partial(I... i) const {
    if constexpr (sizeof...(i) == 0)
      return value_;
    else
      return derivatives_[derivatives_.derivativeIndex(i...)];
  }

  /**
   * \brief Obtain `i...` partial derivative
   *
   * `adValue.partial(i,j, ...)` returns the partial
   * derivative in directions `i,j,...`. Passing no
   * arguments returns the value.
   */
  template <class... I> auto &partial(I... i) {
    if constexpr (sizeof...(i) == 0)
      return value_;
    else
      return derivatives_[derivatives_.derivativeIndex(i...)];
  }

  /**
   * \brief Const cast to `Value`
   *
   * This is equivalent to `adValue.value()`.
   */
  explicit operator Value &() const { return value_; }

  /**
   * \brief Mutable cast to `Value`
   *
   * This is equivalent to `adValue.value()`.
   */
  explicit operator const Value &() const { return value_; }

  // Arithmetic operators to make this behave like a raw Value object

  // Sign operators

  ADValue operator+() const { return *this; }

  ADValue operator-() const { return -1 * (*this); }

  // Binary operators for an ADValue and another type

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

  template <class Other>
  friend ADValue operator/(const ADValue &x, const Other &y) {
    auto z = x;
    z /= y;
    return z;
  }

  template <class Other>
  friend ADValue operator/(const Other &x, const ADValue &y) {
    return x * inv(y);
  }

  // Compound assignment from another type

  template <class Other> ADValue &operator+=(const Other &y) {
    value_ += y;
    return *this;
  }

  template <class Other> ADValue &operator-=(const Other &y) {
    value_ -= y;
    return *this;
  }

  template <class Other> ADValue &operator*=(const Other &y) {
    value_ *= y;
    for (auto k : range(derivatives_.size()))
      derivatives_[k] *= y;
    return *this;
  }

  template <class Other> ADValue &operator/=(const Other &y) {
    value_ /= y;
    for (auto k : range(derivatives_.size()))
      derivatives_[k] /= y;
    return *this;
  }

  // Binary operators with two ADValues

  friend ADValue operator+(const ADValue &x, const ADValue &y) {
    if constexpr (dim != dynamicDim) {
      auto z = x;
      z += y;
      return z;
    } else {
      auto z = Impl::createWithMaxDimension(x, y);
      auto X = Impl::paddedDerivativeAccess(x);
      auto Y = Impl::paddedDerivativeAccess(y);
      auto Z = Impl::storedDerivativeAccess(z);
      Z() = X() + Y();
      if constexpr (maxOrder >= 1)
        for (auto i : range(z.derivatives_.dimension())) {
          Z(i) = X(i) + Y(i);
          if constexpr (maxOrder >= 2)
            for (auto j : range(z.derivatives_.storedDerivatives(i)))
              Z(i, j) = X(i, j) + Y(i, j);
        }
      return z;
    }
  }

  friend ADValue operator-(const ADValue &x, const ADValue &y) {
    if constexpr (dim != dynamicDim) {
      auto z = x;
      z -= y;
      return z;
    } else {
      auto z = Impl::createWithMaxDimension(x, y);
      auto X = Impl::paddedDerivativeAccess(x);
      auto Y = Impl::paddedDerivativeAccess(y);
      auto Z = Impl::storedDerivativeAccess(z);
      Z() = X() - Y();
      if constexpr (maxOrder >= 1)
        for (auto i : range(z.derivatives_.dimension())) {
          Z(i) = X(i) - Y(i);
          if constexpr (maxOrder >= 2)
            for (auto j : range(z.derivatives_.storedDerivatives(i)))
              Z(i, j) = X(i, j) - Y(i, j);
        }
      return z;
    }
  }

  friend ADValue operator*(const ADValue &x, const ADValue &y) {
    auto z = Impl::createWithMaxDimension(x, y);
    auto X = Impl::paddedDerivativeAccess(x);
    auto Y = Impl::paddedDerivativeAccess(y);
    auto Z = Impl::storedDerivativeAccess(z);
    Z() = X() * Y();
    if constexpr (maxOrder >= 1)
      for (auto i : range(z.derivatives_.dimension())) {
        Z(i) = X(i) * Y() + X() * Y(i);
        if constexpr (maxOrder >= 2)
          for (auto j : range(z.derivatives_.storedDerivatives(i)))
            Z(i, j) = X() * Y(i, j) + Y() * X(i, j) + X(i) * Y(j) + X(j) * Y(i);
      }
    return z;
  }

  friend ADValue operator/(const ADValue &x, const ADValue &y) {
    auto z = Impl::createWithMaxDimension(x, y);
    auto X = Impl::paddedDerivativeAccess(x);
    auto Y = Impl::paddedDerivativeAccess(y);
    auto Z = Impl::storedDerivativeAccess(z);
    Z() = X() / Y();
    if constexpr (maxOrder >= 1) {
      auto Y_squared = Y() * Y();
      for (auto i : range(z.derivatives_.dimension())) {
        Z(i) = (X(i) * Y() - X() * Y(i)) / Y_squared;
        if constexpr (maxOrder >= 2) {
          auto Y_cubed = Y_squared * Y();
          for (auto j : range(z.derivatives_.storedDerivatives(i)))
            Z(i, j) =
                (X(i, j) * Y_squared - X() * Y(i, j) * Y() +
                 2 * X() * Y(i) * Y(j) - (X(i) * Y(j) + X(j) * Y(i)) * Y()) /
                Y_cubed;
        }
      }
    }
    return z;
  }

  // Compound assignment from an ADValue

  ADValue &operator+=(const ADValue &y) {
    derivatives_.updateDimension(y.derivatives_);
    auto X = Impl::storedDerivativeAccess(*this);
    auto Y = Impl::storedDerivativeAccess(y);
    X() += Y();
    if constexpr (maxOrder >= 1)
      for (auto i : range(y.derivatives_.dimension())) {
        X(i) += Y(i);
        if constexpr (maxOrder >= 2)
          for (auto j : range(y.derivatives_.storedDerivatives(i)))
            X(i, j) += Y(i, j);
      }
    return *this;
  }

  ADValue &operator-=(const ADValue &y) {
    derivatives_.updateDimension(y.derivatives_);
    auto X = Impl::storedDerivativeAccess(*this);
    auto Y = Impl::storedDerivativeAccess(y);
    X() -= Y();
    if constexpr (maxOrder >= 1)
      for (auto i : range(y.derivatives_.dimension())) {
        X(i) -= Y(i);
        if constexpr (maxOrder >= 2)
          for (auto j : range(y.derivatives_.storedDerivatives(i)))
            X(i, j) -= Y(i, j);
      }
    return *this;
  }

  ADValue &operator*=(const ADValue &y) {
    derivatives_.updateDimension(y.derivatives_);
    auto X = Impl::storedDerivativeAccess(*this);
    auto Y = Impl::storedDerivativeAccess(y);
    if constexpr (maxOrder >= 2) {
      for (auto i : range(y.derivatives_.dimension())) {
        for (auto j : range(y.derivatives_.storedDerivatives(i)))
          X(i, j) = X() * Y(i, j) + Y() * X(i, j) + X(i) * Y(j) + X(j) * Y(i);
        for (auto j : range(y.derivatives_.storedDerivatives(i),
                            derivatives_.storedDerivatives(i)))
          X(i, j) = Y() * X(i, j) + X(j) * Y(i);
      }
      for (auto i :
           range(y.derivatives_.dimension(), derivatives_.dimension())) {
        for (auto j : range(y.derivatives_.dimension()))
          X(i, j) = Y() * X(i, j) + X(i) * Y(j);
        for (auto j : range(y.derivatives_.dimension(),
                            derivatives_.storedDerivatives(i)))
          X(i, j) = Y() * X(i, j);
      }
    }
    if constexpr (maxOrder >= 1) {
      for (auto i : range(y.derivatives_.dimension()))
        X(i) = X(i) * Y() + X() * Y(i);
      for (auto i : range(y.derivatives_.dimension(), derivatives_.dimension()))
        X(i) = X(i) * Y();
    }
    X() = X() * Y();
    return *this;
  }

  ADValue &operator/=(const ADValue &y) {
    (*this) *= inv(y);
    return *this;
  }

private:
  Value value_;
  DerivativeStorage derivatives_;
};

/**
 * \brief Helper function for implementing `ADValue`-aware nonlinear functions.
 * \ingroup TapeLess
 * \param x Inner value, this can be an `ADValue` or a raw value
 * \param f Outer function
 *
 * This computes the composition with custom nonlinear function.
 * The function `f` should implement derivatives of order `k`
 * using `f(std::integral_constant<std::size_t, k>(), x)`.
 * If `x` is a raw value, this simply returns
 * `f(std::integral_constant<std::size_t, 0>(), x)`.
 * If `x` is an `ADValue`, then the return value is obtained
 * by composition of the inner function represented by `x`
 * with `f`, i.e. the partial derivatives are set using the chain rule.
 * Currently only derivatives up to order two are supported.
 */
template <class Value, class Derivatives>
auto adCompose(const Value &x, const Derivatives &f) {
  auto _0 = std::integral_constant<std::size_t, 0>();
  auto _1 = std::integral_constant<std::size_t, 1>();
  auto _2 = std::integral_constant<std::size_t, 2>();
  if constexpr (IsADValue_t<Value>) {
    // (f o x)'  = f'(x)*x'
    // (f o x)'' = f''(x)*x'*x' + f'(x)*x''
    auto y = x;
    auto X = Impl::storedDerivativeAccess(x);
    auto Y = Impl::storedDerivativeAccess(y);
    Y() = f(_0, X());
    if constexpr (Value::maxOrder >= 1) {
      auto df_x = f(_1, X());
      for (auto i : range(y.derivativeStorage().dimension()))
        Y(i) *= df_x;
      if constexpr (Value::maxOrder >= 2) {
        auto ddf_x = f(_2, X());
        for (auto i : range(y.derivativeStorage().dimension()))
          for (auto j : range(y.derivativeStorage().storedDerivatives(i)))
            Y(i, j) = X(i, j) * df_x + ddf_x * X(i) * X(j);
      }
    }
    return y;
  } else
    return f(_0, x);
}

// Some scalar mathematical functions

/**
 * \brief `ADValue`-aware nonlinear reciprocal function
 * \ingroup TapeLess
 */
template <class X> X inv(const X &x) {
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    constexpr auto canUseInt = requires() { 1 / x; };
    using Constant = std::conditional_t<canUseInt, int, decltype(x)>;
    if constexpr (order == 0)
      return Constant(1) / x;
    if constexpr (order == 1)
      return Constant(-1) / (x * x);
    if constexpr (order == 2)
      return Constant(2) / (x * x * x);
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief `ADValue`-aware fabs-function
 * \ingroup TapeLess
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
 * \brief `ADValue`-aware abs-function
 * \ingroup TapeLess
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
 * \brief `ADValue`-aware sin-function
 * \ingroup TapeLess
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
 * \brief `ADValue`-aware cos-function
 * \ingroup TapeLess
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
 * \brief `ADValue`-aware exp-function
 * \ingroup TapeLess
 */
template <class X> auto exp(const X &x) {
  using std::exp;
  return adCompose(x, [](auto order, auto x) { return exp(x); });
}

/**
 * \brief `ADValue`-aware log-function
 * \ingroup TapeLess
 */
template <class X> auto log(const X &x) {
  using std::log;
  using std::pow;
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    constexpr auto canUseInt = requires() { 1 / x; };
    using Constant = std::conditional_t<canUseInt, int, decltype(x)>;
    if constexpr (order == 0)
      return log(x);
    if constexpr (order == 1)
      return Constant(1) / x;
    if constexpr (order == 2)
      return Constant(-1) / (x * x);
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief `ADValue`-aware sqrt-function
 * \ingroup TapeLess
 */
template <class X> auto sqrt(const X &x) {
  using std::sqrt;
  return adCompose(x, [](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    constexpr auto canUseDouble = requires() { 1. / x; };
    using Constant = std::conditional_t<canUseDouble, double, decltype(x)>;
    if constexpr (order == 0)
      return sqrt(x);
    if constexpr (order == 1)
      return Constant(1. / 2.) / sqrt(x);
    if constexpr (order == 2)
      return Constant(-1. / 4.) / (x * sqrt(x));
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief `ADValue`-aware pow-function with constant exponent
 * \ingroup TapeLess
 */
template <class X, class Y, std::enable_if_t<not IsADValue_t<Y>, int> = 0>
auto pow(const X &x, const Y &y) {
  using std::pow;
  return adCompose(x, [y](auto k, auto x) {
    constexpr auto order = decltype(k)::value;
    constexpr auto canUseInt = requires() { y - 1; };
    using Constant = std::conditional_t<canUseInt, int, decltype(y)>;
    if constexpr (order == 0)
      return pow(x, y);
    if constexpr (order == 1)
      return y * pow(x, y - Constant(1));
    if constexpr (order == 2)
      return y * (y - Constant(1)) * pow(x, y - Constant(2));
    static_assert(order <= 2, "Only derivatives up to order 2 are implemented");
  });
}

/**
 * \brief `ADValue`-aware pow-function with constant base
 * \ingroup TapeLess
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
 * \brief `ADValue`-aware pow-function variable base and exponent
 * \ingroup TapeLess
 */
template <class V, std::enable_if_t<IsADValue_t<V>, int> = 0>
auto pow(const V &x, const V &y) {
  using std::log;
  using std::pow;
  auto z = Impl::createWithMaxDimension(x, y);
  auto X = Impl::paddedDerivativeAccess(x);
  auto Y = Impl::paddedDerivativeAccess(y);
  auto Z = Impl::storedDerivativeAccess(z);
  // Here we use:
  // z  = pow(x, y) = exp(y*log(x))
  // z' = exp(y*log(x)) * (y'*log(x) + y/x*x') = y*x^(y-1)*x' + log(x)*z*y'
  Z() = pow(X(), Y());
  if constexpr (V::maxOrder >= 1) {
    auto pow_X_Y1 = pow(X(), Y() - 1);
    auto log_X = log(X());
    auto tmp1 = Y() * pow_X_Y1;
    auto tmp2 = Z() * log_X;
    for (auto i : range(z.derivativeStorage().dimension()))
      Z(i) = tmp1 * X(i) + tmp2 * Y(i);
    if constexpr (V::maxOrder >= 2) {
      auto tmp3 = Y() * (Y() - 1) * pow(X(), Y() - 2);
      auto tmp4 = tmp2 * log_X;
      auto tmp5 = (1 + log_X * Y()) * pow_X_Y1;
      for (auto i : range(z.derivativeStorage().dimension()))
        for (auto j : range(z.derivativeStorage().storedDerivatives(i)))
          Z(i, j) = tmp1 * X(i, j) + tmp2 * Y(i, j) + tmp3 * X(i) * X(j) +
                    tmp4 * Y(i) * Y(j) + tmp5 * (X(i) * Y(j) + Y(i) * X(j));
    }
  }
  return z;
}

} // end namespace adolc

#endif // DUNE_FUFEM_FUNCTIONS_ADVALUE_HH
