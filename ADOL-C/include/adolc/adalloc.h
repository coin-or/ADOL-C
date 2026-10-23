/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adalloc.h
 Revision: $Id$
 Contents: Allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#if !defined(ADOLC_ADALLOC_H)
#define ADOLC_ADALLOC_H 1

#include <adolc/internal/common.h>
/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS
#include <stddef.h>
/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */
ADOLC_API char *populate_dpp(double ***const pointer, char *const memory,
                             size_t n, size_t m);
ADOLC_API char *populate_dppp(double ****const pointer, char *const memory,
                              size_t n, size_t m, size_t p);
ADOLC_API char *populate_dppp_nodata(double ****const pointer,
                                     char *const memory, size_t n, size_t m);

END_C_DECLS

/****************************************************************************/
/*                                                       Now the C++ THINGS */
#if defined(__cplusplus)

#include <adolc/adolcexport.h>
#include <array>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */

namespace {
template <typename T> T fillIdentity(T identity) {
  for (size_t i = 0; i < identity.shape().first; i++) {
    identity[i][i] = static_cast<typename T::value_type>(1.0);
  }
  return identity;
}
} // namespace

/**
 * @brief Creates a view in matrix shape of std::array
 *
 * Creates of view in matrix shape of an std::array, i.e.
 * for a vector ( 1 2 3 4 ) and given dimension (2, 2) it can
 * provide a view, that behaves like the following matrix:
 * ( 1 2 )
 * ( 3 4 )
 *
 * @tparam rows_ Number of output dimension (rows).
 * @tparam cols_ Number of input dimension (columns).
 * @tparam T    Datatype of the content.
 * @param owner     Reference to Container that holds the matrix's
 *                  content.
 * @return std::array, which holds pointers to the subsections of the
 *         input array, and splits it into rows.
 */
template <size_t rows_, size_t cols_, typename T>
std::array<T *, rows_> matrixView(std::array<T, rows_ * cols_> &owner) {
  assert(rows_ * cols_ == owner.size());
  std::array<T *, rows_> view;
  for (size_t i = 0; i < rows_; i++) {
    view[i] = owner.data() + i * cols_;
  }
  return view;
}
/**
 * @brief Creates a view in matrix shape of std::vector-like containers.
 *
 * Creates of view in matrix shape of an std::vector-like container, i.e.
 * for a vector ( 1 2 3 4 ) and given dimension (2, 2) it can
 * provide a view, that behaves like the following matrix:
 * ( 1 2 )
 * ( 3 4 )
 *
 * @tparam T Datatype of the content.
 * @tparam VS Optional further template parameter
 * @tparam Cont Container type.
 * @param owner     Reference to Container that holds the matrix's
 *                  content.
 * @param rows      Number of rows of the matrix.
 * @param cols      Number of columns of the matrix.
 * @return Container – of same type as the input –, which holds poin-
 *         ters to the subsections of the input container, and splits
 *         it into rows.
 * @note VS is passed to the Container directly. If one of the parameter V of VS
 * depends on the first argument T* problems can occur if V was deduced with T
 */
template <typename T, typename... VS,
          template <typename, typename...> typename Cont>
Cont<T *, VS...> matrixView(Cont<T, VS...> &owner, size_t rows, size_t cols) {
  assert(rows * cols == owner.size());
  Cont<T *, VS...> view(rows);
  for (size_t i = 0; i < rows; i++) {
    view[i] = owner.data() + i * cols;
  }
  return view;
}

/// @brief Specialized version of matrixView that handles the std::vector
/// template properly
template <typename T, template <typename> typename Allocator>
std::vector<T *, Allocator<T *>> matrixView(std::vector<T, Allocator<T>> &owner,
                                            size_t rows, size_t cols) {
  assert(rows * cols == owner.size());
  std::vector<T *, Allocator<T *>> view(rows);
  for (size_t i = 0; i < rows; i++) {
    view[i] = owner.data() + i * cols;
  }
  return view;
}

/**
 * @brief Adol-C's own Matrix-like containers.
 *
 *
 * @tparam Data type. Defaults to double.
 * @tparam Rowsize. Defaults to zero.
 * @tparam Colsize. Defaults to Rowsize.
 */
template <typename T = double, size_t rows_ = std::dynamic_extent,
          size_t cols_ = rows_>
class Matrix {
  static_assert(cols_ > 0 && rows_ > 0);
  std::array<T *, rows_> matrix_{};     /// Splits the data into row vectors.
  std::array<T, rows_ * cols_> data_{}; /// Holds the matrix's content.

  constexpr void populate() {
    for (size_t i = 0; i < matrix_.size(); i++) {
      matrix_[i] = data_.data() + (i * cols_);
    }
  }

public:
  using value_type = T;
  ~Matrix() = default;

  /**
   * @brief Constructor for Matrix-object
   *
   * Allocates the matrix content contiguously as a an std::array
   * and allocates a second std::array with pointers to parts of
   * the data_ to split the content into a matrix format.
   *
   */
  constexpr Matrix() { populate(); }

  /**
   * @brief Constructor for Matrix-object
   *
   * Allocates the matrix content contiguously as a an std::array
   * and allocates a second std::array with pointers to parts of
   * the data_ to split the content into a matrix format.
   *
   * @param filler    Value to fill the matrix with
   */
  constexpr Matrix(value_type filler) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = filler;
    }
    populate();
  }

  constexpr Matrix(const Matrix &other) : Matrix() { data_ = other.data_; }
  constexpr Matrix &operator=(const Matrix &other) {
    data_ = other.data_;
    return *this;
  }

  constexpr Matrix(Matrix &&other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : matrix_(std::move(other.matrix_)), data_(std::move(other.data_)) {
    populate();
  }

  constexpr Matrix &
  operator=(Matrix &&other) noexcept(std::is_nothrow_move_assignable_v<T>) {
    if (this != &other) {
      data_ = std::move(other.data_);
      populate();
    }
    return *this;
  }
  /**
   * @brief std::span() conversion overload.
   *
   * Useful for functions calls, e.g. fov_reverse().
   *
   */
  constexpr operator std::span<value_type *>() {
    return std::span<value_type *>(matrix_);
  }
  constexpr operator std::span<value_type *const>() {
    return std::span<value_type *const>(matrix_);
  }
  /**
   * @brief Enable row-access of Matrix container.
   * @overload []-operator
   *
   * Jump to virtual row of the Matrix.data_ vector.
   *
   * @param row       Row, that is supposed to be jumped to.
   */
  constexpr value_type *operator[](size_t row) {
    assert(row < rows_);
    return matrix_[row];
  }
  constexpr const value_type *operator[](size_t row) const {
    assert(row < rows_);
    return matrix_[row];
  }
  /**
   * @brief Retrieving front of the matrix.
   *
   * Here for symmetric functionality to iterables like
   * std::vector.
   */
  constexpr value_type **data() { return matrix_.data(); }
  constexpr const value_type *const *data() const { return matrix_.data(); }
  /**
   * @brief Retrieves dimensions of Matrix-object.
   */
  constexpr std::pair<size_t, size_t> shape() const { return {rows_, cols_}; }
  /**
   * @brief Fills entire Matrix Container with given value.
   *
   *@param filler     Value to fill the matrix with.
   */
  constexpr void fill(value_type filler) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = filler;
    }
  }
};

/**
 * @brief Partial Specialization of Adol-C's own Matrix-like containers.
 *
 * This implementation uses std::vector as containers.
 *
 * @tparam Data type. Defaults to double.
 */
template <typename T>
class Matrix<T, std::dynamic_extent, std::dynamic_extent> {
  std::vector<T *> matrix_{}; /// Splits the data into row vectors.
  std::vector<T> data_{};     /// Holds the matrix's content.
  size_t rows_{0};
  size_t cols_{0};

  constexpr void populate() {
    for (size_t i = 0; i < rows_; i++) {
      matrix_[i] = data_.data() + (i * cols_);
    }
  }

public:
  using value_type = T;
  ~Matrix() = default;

  constexpr Matrix() = default;
  /**
   * @brief Constructor for Matrix-object
   *
   * Allocates the matrix content contiguously as a an std::vector
   * and allocates a second std::vector with pointers to parts of
   * the data_ to split the content into a matrix format.
   *
   * @param rows      Number of rows of the matrix.
   * @param cols      Number of columns of the matrix.
   */
  constexpr Matrix(size_t rows, size_t cols, value_type filler = 0.0)
      : matrix_(rows), data_(rows * cols, filler), rows_(rows), cols_(cols) {
    assert(rows_ > 0 && cols_ > 0 && "Zero dimension are not allowed!");
    populate();
  }

  /**
   * @brief Creates a Square Matrix.
   *
   * @param rowsAndCols Number of rows and columns of the matrix.
   */
  constexpr explicit Matrix(size_t rowsAndCols)
      : Matrix(rowsAndCols, rowsAndCols) {}
  /**
   * @brief Constructs from pre-existing objects outside of Matrix()
   *
   * Moves a Matrix-like object specified independently
   * outisde of the class into a Matrix object.
   *
   * @param rows      Data view.
   * @param data      Matrix data.
   * @return Moved Matrix-object.
   */
  constexpr Matrix(std::vector<T *> &&matrix, std::vector<T> &&data)
      : matrix_(std::move(matrix)), data_(std::move(data)),
        rows_(matrix_.size()),
        cols_(matrix_.size() > 0 ? data_.size() / matrix_.size() : 0) {
    assert(rows_ * cols_ == data_.size());
    populate();
  }

  constexpr Matrix(const Matrix &other)
      : matrix_(other.rows_), data_(other.data_), rows_(other.rows_),
        cols_(other.cols_) {
    populate();
  }

  constexpr Matrix &operator=(const Matrix &other) {
    assert(rows_ == other.rows_ && cols_ == other.cols_);
    data_ = other.data_;
    return *this;
  }

  constexpr Matrix(Matrix &&other) noexcept
      : matrix_(std::move(other.matrix_)), data_(std::move(other.data_)),
        rows_(std::exchange(other.rows_, 0)),
        cols_(std::exchange(other.cols_, 0)) {}

  constexpr Matrix &operator=(Matrix &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      matrix_ = std::move(other.matrix_);
      rows_ = std::exchange(other.rows_, 0);
      cols_ = std::exchange(other.cols_, 0);
      populate();
    }
    return *this;
  }
  /**
   * @brief std::span() conversion overload.
   *
   * Useful for functions calls, e.g. fov_reverse().
   *
   */
  constexpr operator std::span<value_type *>() {
    return std::span<value_type *>(matrix_);
  }
  constexpr operator std::span<value_type *const>() {
    return std::span<value_type *const>(matrix_);
  }
  /**
   * @brief Enable row-access of Matrix container.
   * @overload []-operator
   *
   * Jump to virtual row of the Matrix.data_ std::vector.
   *
   * @param row       Row, that is supposed to be jumped to.
   */
  constexpr value_type *operator[](size_t row) {
    assert(row < rows_);
    return matrix_[row];
  }
  constexpr const value_type *operator[](size_t row) const {
    assert(row < rows_);
    return matrix_[row];
  }
  /**
   * @brief Retrieving front of the matrix.
   *
   * Here for symmetric functionality to iterables like
   * std::vector.
   */
  constexpr value_type **data() { return matrix_.data(); }
  constexpr const value_type *const *data() const { return matrix_.data(); }
  /**
   * @brief Retrieves dimensions of Matrix-object.
   */
  constexpr std::pair<size_t, size_t> shape() const { return {rows_, cols_}; }
  /**
   * @brief Fills entire Matrix Container with given value.
   *
   *@param filler     Value to fill the matrix with.
   */
  constexpr void fill(value_type filler = 0.0) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = filler;
    }
  }
};

/**
 * @brief Adol-C's own 3D-Tensor-like containers.
 *
 * This implementation uses std::vector as containers.
 *
 * This class can be used to make containers that are structured like
 * three dimensional tensor and can be accessed as such.
 *
 * It does NOT come with perhaps expected arithmetic capabilites, e.g.
 * tensor-addition.
 *
 * @tparam Data type. Defaults to double.
 */
template <typename T = double> class Tensor {
  std::vector<T **> tensor_{};
  std::vector<T *> slices_{};
  std::vector<T> data_{};
  size_t rows_{0};
  size_t cols_{0};
  size_t depths_{0};

  constexpr void populate() {
    for (size_t i = 0; i < rows_ * cols_; i++) {
      slices_[i] = data_.data() + i * depths_;
    }
    for (size_t i = 0; i < rows_; i++) {
      tensor_[i] = slices_.data() + i * cols_;
    }
  }

public:
  using value_type = T;
  ~Tensor() = default;
  constexpr Tensor() = default;
  /**
   * @brief Constructor for Tensor-object
   *
   * Allocates the tensor content contiguously as a an std::vector
   * and allocates a additional std::vectors with pointers to parts of
   * the data_ to split the content into a tensor.
   *
   * @param rows      Number of rows of the tensor.
   * @param cols      Number of columns of the tensor.
   * @param dep       Size of the third axis (depth) of the tensor.
   */
  constexpr Tensor(size_t rows, size_t cols, size_t depths,
                   value_type filler = 0.0)
      : tensor_(rows), slices_(rows * cols),
        data_(rows * cols * depths, filler), rows_(rows), cols_(cols),
        depths_(depths) {
    assert(rows_ > 0 && cols_ > 0 && depths_ > 0 &&
           "Zero dimension are not allowed!");
    populate();
  }

  constexpr Tensor(const Tensor &other)
      : tensor_(other.rows_), slices_(other.rows_ * other.cols_),
        data_(other.data_), rows_(other.rows_), cols_(other.cols_),
        depths_(other.depths_) {
    populate();
  }

  constexpr Tensor &operator=(const Tensor &other) {
    assert(rows_ == other.rows_ && cols_ == other.cols_ &&
           depths_ == other.depths_);
    data_ = other.data_;
    return *this;
  }

  constexpr Tensor(Tensor &&other) noexcept
      : tensor_(std::move(other.tensor_)), slices_(std::move(other.slices_)),
        data_(std::move(other.data_)), rows_(std::exchange(other.rows_, 0)),
        cols_(std::exchange(other.cols_, 0)),
        depths_(std::exchange(other.depths_, 0)) {
    populate();
  }

  constexpr Tensor &operator=(Tensor &&other) noexcept {
    if (this != &other) {
      data_ = std::move(other.data_);
      slices_ = std::move(other.slices_);
      tensor_ = std::move(other.tensor_);
      rows_ = std::exchange(other.rows_, 0);
      cols_ = std::exchange(other.cols_, 0);
      depths_ = std::exchange(other.depths_, 0);
      populate();
    }
    return *this;
  }
  /**
   * @brief std::span() conversion overload.
   *
   * Useful for functions calls, e.g. fov_reverse().
   *
   */
  /* operator value_type ***() { return tensor_.data(); } */
  constexpr operator std::span<value_type **>() {
    return std::span<value_type **>(tensor_);
  }
  constexpr operator std::span<value_type **const>() {
    return std::span<value_type **const>(tensor_);
  }
  /**
   * @brief Enable row-access of Tensor container.
   * @overload []-operator
   *
   * Jump to virtual row of the Tensor.data_ array.
   *
   * @param row       Row, that is supposed to be jumped to.
   */
  constexpr value_type **operator[](size_t row) {
    assert(row < rows_);
    return tensor_[row];
  }
  constexpr const value_type *const *operator[](size_t row) const {
    assert(row < rows_);
    return tensor_[row];
  }
  /**
   * @brief Retrieving front of the Tensor.
   *
   * Here for symmetric functionality to iterables like
   * std::vector.
   */
  constexpr value_type ***data() { return tensor_.data(); }
  constexpr const value_type *const *const *data() const {
    return tensor_.data();
  }
  /**
   * @brief Fills entire Tensor Container with given value.
   *
   *@param filler     Value to fill the tensor with.
   */
  /**
   * @brief Retrieves dimensions of Tensor-object.
   */
  constexpr std::array<size_t, 3> shape() const {
    return {rows_, cols_, depths_};
  }
  constexpr void fill(value_type filler) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = filler;
    }
  }
};

/**
 * @brief Unit Vectors for different containers.
 *
 * Used to make unit vectors for user-specified containers that are
 * std::vector-like.
 *
 * @tparam T Container type.
 * @param dim      Dimension of vector.
 * @param dir      Unit  direction.
 * @return Unit vector as specified output container.
 */
template <typename T = double,
          template <typename, typename...> typename Cont = std::vector>
constexpr Cont<T> unitVector(size_t dim, size_t dir) {
  assert(dir <= dim);
  Cont<T> unitVec(dim, T{});
  unitVec[dir - 1] = static_cast<T>(1.0);
  return unitVec;
}
/**
 * @brief Unit Vector as std::array.
 *
 * @tparam T   Data type.
 * @tparam dim Dimension of the vector.
 * @param dir      Unit  direction.
 * @return Unit vector as std::array.
 */
template <typename T = double, size_t dim>
constexpr std::array<T, dim> unitVector(size_t dir) {
  assert(dir <= dim);
  std::array<T, dim> unitVec{};
  unitVec[dir - 1] = static_cast<T>(1.0);
  return unitVec;
}

/**
 * @brief Square Identity Matrix (Stack)
 *
 * Returns a Matrix-Container object, which is allocated
 * (using std::array) on the stack.
 *
 * @tparam dim  Input/Output dimension.
 * @tparam T    data type.
 * @return Matrix of size Dim x Dim.
 */
template <typename T = double, size_t dim = 1>
constexpr Matrix<T, dim, dim> unitMatrix() {
  auto identity = Matrix<T, dim, dim>{};
  return fillIdentity(identity);
}
/**
 * @brief Square Identity Matrix (Heap)
 *
 * Returns a Matrix-Container object, which is generally allocated
 * (using std::vector) on the heap.
 *
 * @param dim  Input/Output dimension.
 * @return Matrix of size Dim x Dim.
 */
template <typename T = double>
constexpr Matrix<T, std::dynamic_extent, std::dynamic_extent>
unitMatrix(size_t dim) {
  auto identity = Matrix<T, std::dynamic_extent, std::dynamic_extent>{dim};
  return fillIdentity(identity);
}

#endif // _cplusplus

/****************************************************************************/
#endif // ADOLC_ADALLOC_H
