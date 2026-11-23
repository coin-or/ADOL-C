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
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ADOLC {

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */

/**
 * @brief View an array as rows of contiguous values.
 * @pre The owner must outlive all accesses through the returned row pointers.
 * @return Row pointers into the original array. Writes update the array.
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
 * @brief View contiguous storage as a matrix with the given row and column
 * counts.
 * @pre The owner has exactly `rows * cols` entries in row-major order.
 * @return Row pointers that borrow the owner's storage. Rebuild the view if
 * the owner reallocates.
 * @note This overload reuses the container's extra template arguments for the
 * row table. They must also work with pointer elements.
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

/// @brief Borrow rows from a vector of exactly `rows * cols` entries.
/// Keep the vector's storage alive and at the same address while using the
/// rows.
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
 * @brief Dense matrix that owns contiguous row-major values and a table of row
 * pointers.
 *
 * Omit both extents to construct `Matrix<T>(rows, cols)` at run time.
 *
 * @tparam T Element type (default `double`).
 * @tparam rows_ Positive fixed row count, or `std::dynamic_extent` for runtime
 * dimensions.
 * @tparam cols_ Fixed column count. Defaults to `rows_`, so `Matrix<T, N>` is
 * square.
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
   * @brief Value-initialize every entry of the fixed-size matrix.
   */
  constexpr Matrix() { populate(); }

  /**
   * @brief Initialize every entry to `filler`.
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
  constexpr operator std::span<const value_type *const>() const {
    return {matrix_.data(), matrix_.size()};
  }
  /// @brief Return the start of the zero-based row.
  constexpr value_type *operator[](size_t row) {
    assert(row < rows_);
    return matrix_[row];
  }
  constexpr const value_type *operator[](size_t row) const {
    assert(row < rows_);
    return matrix_[row];
  }
  /// @brief Return the row-pointer table for drivers that accept raw
  /// pointers.
  constexpr value_type **data() { return matrix_.data(); }
  constexpr const value_type *const *data() const { return matrix_.data(); }
  /**
   * @brief Return the row and column counts, in that order.
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
 * @brief Dense matrix with runtime dimensions and contiguous row-major
 * values. A default-constructed matrix is empty.
 * @tparam T Element type.
 */
template <typename T>
class Matrix<T, std::dynamic_extent, std::dynamic_extent> {
  std::vector<T *> matrix_{}; /// Splits the data into row vectors.
  std::vector<T> data_{};     /// Holds the matrix's content.
  size_t rows_{0};
  size_t cols_{0};

  static constexpr std::size_t checkedDimensions(std::size_t rows,
                                                 std::size_t cols) {
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Dimensions must be positive");
    }
    return rows;
  }

  constexpr void populate() {
    assert(rows_ == matrix_.size());
    assert(rows_ * cols_ == data_.size());
    for (size_t i = 0; i < rows_; i++) {
      matrix_[i] = data_.data() + (i * cols_);
    }
  }

public:
  using value_type = T;
  ~Matrix() = default;

  constexpr Matrix() = default;
  /**
   * @brief Allocate a matrix with positive row and column counts.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param filler Initial value for every entry (defaults to zero).
   */
  constexpr Matrix(size_t rows, size_t cols, value_type filler = 0.0)
      : matrix_(checkedDimensions(rows, cols)), data_(rows * cols, filler),
        rows_(rows), cols_(cols) {
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
   * @brief Take ownership of a row table and contiguous row-major values.
   * @param matrix Row table whose size determines the row count. Row pointers
   * are rebuilt.
   * @param data Values divided evenly among the rows. An empty row table
   * requires empty data.
   */
  constexpr Matrix(std::vector<T *> &&matrix, std::vector<T> &&data)
      : matrix_(std::move(matrix)), data_(std::move(data)),
        rows_(matrix_.size()),
        cols_(matrix_.size() > 0 ? data_.size() / matrix_.size() : 0) {
    populate();
  }

  constexpr Matrix(const Matrix &other)
      : matrix_(other.rows_), data_(other.data_), rows_(other.rows_),
        cols_(other.cols_) {
    populate();
  }

  /// @pre Both matrices have the same shape.
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
  constexpr operator std::span<const value_type *const>() const {
    return {matrix_.data(), matrix_.size()};
  }
  /// @brief Return the start of the zero-based row.
  constexpr value_type *operator[](size_t row) {
    assert(row < rows_);
    return matrix_[row];
  }
  constexpr const value_type *operator[](size_t row) const {
    assert(row < rows_);
    return matrix_[row];
  }
  /// @brief Return the row-pointer table for drivers that accept raw
  /// pointers.
  constexpr value_type **data() { return matrix_.data(); }
  constexpr const value_type *const *data() const { return matrix_.data(); }
  /**
   * @brief Return the row and column counts, in that order.
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
 * @brief Owns a contiguous three-dimensional array with runtime dimensions.
 * @tparam T Element type.
 */
template <typename T = double> class Tensor {
  std::vector<T **> tensor_{};
  std::vector<T *> slices_{};
  std::vector<T> data_{};
  size_t rows_{0};
  size_t cols_{0};
  size_t depths_{0};

  static constexpr std::size_t
  checkedDimensions(std::size_t rows, std::size_t cols, std::size_t depths) {
    if (rows == 0 || cols == 0 || depths == 0) {
      throw std::invalid_argument("Dimensions must be positive");
    }
    return rows;
  }
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
   * @brief Allocate a tensor and initialize every entry to `filler`.
   * @param rows Number of rows.
   * @param cols Number of columns.
   * @param depths Number of entries along the third axis.
   * @param filler Initial value for every entry.
   */
  constexpr Tensor(size_t rows, size_t cols, size_t depths,
                   value_type filler = 0.0)
      : tensor_(checkedDimensions(rows, cols, depths)), slices_(rows * cols),
        data_(rows * cols * depths, filler), rows_(rows), cols_(cols),
        depths_(depths) {
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
  constexpr operator std::span<const value_type *const *const>() const {
    return {tensor_.data(), tensor_.size()};
  }
  /// @brief Return the zero-based row for indexing as
  /// `tensor[row][col][depth]`.
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
  /// @brief Return the row, column, and depth counts in that order.
  constexpr std::array<size_t, 3> shape() const {
    return {rows_, cols_, depths_};
  }
  /// @brief Set every entry to `filler`.
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
  assert(dir < dim);
  Cont<T> unitVec(dim, T{});
  unitVec[dir] = static_cast<T>(1.0);
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
  assert(dir < dim);
  std::array<T, dim> unitVec{};
  unitVec[dir] = static_cast<T>(1.0);
  return unitVec;
}

/**
 * @brief Creates a mutable identity Matrix backed by std::array.
 */
template <typename T = double, size_t dim = 1>
constexpr Matrix<T, dim, dim> unitMatrix() {
  Matrix<T, dim, dim> identity{};
  for (size_t i = 0; i < dim; ++i) {
    identity[i][i] = T{1};
  }
  return identity;
}

/**
 * @brief Creates a mutable, dense identity Matrix backed by std::vector.
 *
 * This allocates O(dim * dim) values. Use Identity<T> for a compact,
 * read-only identity matrix passed to driver inputs.
 */
template <typename T = double> Matrix<T> unitMatrix(size_t dim) {
  Matrix<T> identity(dim, dim);
  for (size_t i = 0; i < dim; ++i) {
    identity[i][i] = T{1};
  }
  return identity;
}

/**
 * @brief Compact read-only identity matrix for driver inputs.
 *
 * For a positive dimension `n`, the rows overlap in a buffer of `2*n - 1`
 * values, sharing a single one for all diagonal entries. The object owns
 * the buffer and its `n` row pointers. Pass it directly as a read-only seed
 * to drivers such as hess_mat(). Use unitMatrix() for a writable dense seed.
 */
template <typename T> class Identity {
  std::vector<T *> matrix_;
  std::vector<T> data_;
  std::size_t dims_{0};

  static constexpr std::size_t checkedDimensions(std::size_t dims) {
    if (dims == 0) {
      throw std::invalid_argument("Identity dimensions must be positive");
    }
    if (dims > std::numeric_limits<std::size_t>::max() / 2 + 1) {
      throw std::length_error("Identity dimensions are too large");
    }
    return dims;
  }

  constexpr void populate() {
    for (size_t i = 0; i < dims_; i++) {
      matrix_[i] = data_.data() + dims_ - 1 - i;
    }
  }

public:
  using value_type = T;
  ~Identity() = default;
  Identity() = delete;
  constexpr explicit Identity(std::size_t dims)
      : matrix_(checkedDimensions(dims)), data_(2 * dims - 1), dims_(dims) {
    data_[dims - 1] = T{1};
    populate();
  }
  Identity(Identity const &other) = delete;
  Identity &operator=(Identity const &other) = delete;

  constexpr Identity(Identity &&other) noexcept
      : matrix_(std::move(other.matrix_)), data_(std::move(other.data_)),
        dims_(std::exchange(other.dims_, 0)) {
    assert(dims_ == matrix_.size());
    populate();
  }

  Identity &operator=(Identity &&other) noexcept {
    if (this != &other) {
      matrix_ = std::move(other.matrix_);
      data_ = std::move(other.data_);
      dims_ = std::exchange(other.dims_, 0);
      assert(dims_ == matrix_.size());
      populate();
    }
    return *this;
  }

  constexpr value_type const *operator[](std::size_t idx) const {
    assert(idx < dims_);
    return matrix_[idx];
  }

  constexpr operator std::span<value_type const *const>() const {
    return {matrix_.data(), matrix_.size()};
  }

  value_type const *const *data() const { return matrix_.data(); }
};

} // namespace ADOLC

#endif // __cplusplus

/****************************************************************************/
#endif // ADOLC_ADALLOC_H
