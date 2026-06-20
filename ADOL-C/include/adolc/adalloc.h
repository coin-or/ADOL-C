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
#include <type_traits>
#if !defined(ADOLC_ADALLOC_H)
#define ADOLC_ADALLOC_H 1

#include <adolc/adolcexport.h>
#include <adolc/internal/common.h>
#include <array>
#include <concepts>
#include <span>
#include <utility>
#include <vector>

/****************************************************************************/
/*                                                         Now the C THINGS */
BEGIN_C_DECLS

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

/*--------------------------------------------------------------------------*/
/*                                              MEMORY MANAGEMENT UTILITIES */

/**
 * @brief Matrix view for std::array containers.
 * @overload of MatrixView().
 *
 * Provides a Matrix view of an std::array – holds the matrix's
 * content contiguously – that behaves like an std::vector, i.e.
 * for a vector ( 1 2 3 4 ) and given dimension (2, 2) it can
 * provide a view, that behaves like the following matrix:
 * ( 1 2 )
 * ( 3 4 )
 *
 * @tparam dimY Number of output dimension (rows).
 * @tparam dimX Number of input dimension (columns).
 * @tparam T    Datatype of the content.
 * @param owner     Reference to Container that holds the matrix's
 *                  content.
 * @param rows      Number of rows of the matrix.
 * @param cols      Number of columns of the matrix.
 * @return std::array, which holds pointers to the subsections of the
 *         input array, and splits it into rows.
 */
ADOLC_API template <size_t dimY, size_t dimX, typename T>
std::array<T *, dimY> MatrixView(std::array<T, dimY * dimX> &owner) {
  assert(dimY * dimX == owner.size());
  std::array<T *, dimY> view;
  for (size_t i = 0; i < dimY; i++) {
    view[i] = owner.data() + i * dimX;
  }
  return view;
}
/**
 * @brief Matrix view for std::vector-like containers.
 *
 * Provides a Matrix view of a container – holds the matrix's con-
 * tent contiguously – that behaves like an std::vector, i.e. for
 * a vector ( 1 2 3 4 ) and given dimension (2, 2) it can provide
 * a view, that behaves like the following matrix:
 * ( 1 2 )
 * ( 3 4 )
 *
 * @tparam T Datatype of the content.
 * @tparam Cont Container type.
 * @param owner     Reference to Container that holds the matrix's
 *                  content.
 * @param rows      Number of rows of the matrix.
 * @param cols      Number of columns of the matrix.
 * @return Container – of same type as the input –, which holds poin-
 *         ters to the subsections of the input container, and splits
 *         it into rows.
 *
 * @overload Works also with std::array with derived notation.
 */
ADOLC_API template <typename T, template <typename, typename...> typename Cont>
Cont<T *> MatrixView(Cont<T> &owner, size_t rows, size_t cols) {
  assert(rows * cols == owner.size());
  Cont<T *> view(rows);
  for (size_t i = 0; i < rows; i++) {
    view[i] = owner.data() + i * cols;
  }
  return view;
}

/**
 * @brief Adol-C's own Matrix-like containers.
 *
 * This class can be used to make containers that are structured like
 * matrices and can be accessed as such.
 *
 * It does NOT come with perhaps expected arithmetic capabilities, e.g.
 * matrix-multiplication.
 *
 * @tparam Rowsize. Defaults to zero.
 * @tparam Colsize. Defaults to Rowsize.
 * @tparam Data type. Defaults to double.
 */
ADOLC_API template <typename T = double, size_t dimY = 0, size_t dimX = dimY>
class Matrix {
  std::array<T *, dimY> matrix_{};    /// Splits the data into row vectors.
  std::array<T, dimY * dimX> data_{}; /// Holds the matrix's content.

public:
  using value_type = T;
  ~Matrix() = default;
  Matrix &operator=(Matrix &&other) = default;
  /**
   * @brief Constructor for Matrix-object
   *
   * Allocates the matrix content contiguously as a an std::array
   * and allocates a second std::vector with pointers to parts of
   * the data_ to split the content into a matrix format.
   *
   * @param rows      Number of rows of the matrix.
   * @param cols      Number of columns of the matrix.
   */
  Matrix(value_type filler = 0.0) {
    assert(dimY > 0 && dimX > 0);
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = filler;
    }
    for (size_t i = 0; i < matrix_.size(); i++) {
      matrix_[i] = data_.data() + (i * dimX);
    }
  }
  /**
   * @brief Copy Constructor for Matrix-object
   * @overload
   *
   * Makes a copy of an already existing Matrix-container with content.
   *
   * @param Matrix    A copy of an Matrix-class object.
   */
  Matrix(const Matrix &other) : Matrix() {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = other.data_[i];
    }
  }
  /**
   * @brief Move Constructor for Matrix-object
   *
   * Moves a matrix.
   *
   * @param other     Matrix to be moved.
   */
  Matrix(Matrix &&other)
      : matrix_(std::move(other.matrix_)), data_(std::move(other.data_)) {
    for (size_t i = 0; i < matrix_.size(); i++) {
      matrix_[i] = data_.data() + (i * dimX);
    }
  }
  /**
   * @brief T**() conversion overload.
   *
   * Useful for functions calls, e.g. fov_reverse().
   *
   */
  /* operator value_type **() { return matrix_.data(); } */
  operator std::span<value_type *>() {
    return std::span<value_type *>(matrix_);
  }
  /**
   * @brief Enable row-access of Matrix container.
   * @overload []-operator
   *
   * Jump to virtual row of the Matrix.data_ vector.
   *
   * @param row       Row, that is supposed to be jumped to.
   */
  value_type *operator[](size_t row) {
    assert(row <= dimY - 1);
    return matrix_[row];
  }
  /**
   * @brief Retrieving front of the matrix.
   *
   * Here for symmetric functionality to iterables like
   * std::vector.
   */
  value_type **data() { return matrix_.data(); }
  /**
   * @brief Retrieves dimensions of Matrix-object.
   */
  std::pair<size_t, size_t> shape() { return {dimY, dimX}; }
  /**
   * @brief Fills entire Matrix Container with given value.
   *
   *@param filler     Value to fill the matrix with.
   */
  void fill(value_type filler = 0.0) {
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
 * This class can be used to make containers that are structured like
 * matrices and can be accessed as such.
 *
 * It does NOT come with perhaps expected arithmetic capabilites, e.g.
 * matrix-multiplication.
 *
 * @tparam Data type. Defaults to double.
 */
ADOLC_API template <typename T> class Matrix<T, 0, 0> {
  std::vector<T *> matrix_{}; /// Splits the data into row vectors.
  std::vector<T> data_{};     /// Holds the matrix's content.

public:
  using value_type = T;
  Matrix() = default;
  ~Matrix() = default;
  Matrix &operator=(Matrix &&other) = default;
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
  Matrix(size_t rows, size_t cols, value_type filler = 0.0)
      : matrix_(rows), data_(rows * cols, filler) {
    assert(rows > 0 && cols > 0);
    for (size_t i = 0; i < rows; i++) {
      matrix_[i] = data_.data() + (i * cols);
    }
  }
  /**
   * @brief Creates a Square Matrix.
   *
   * @param rowsAndCols Number of rows and columns of the matrix.
   */
  Matrix(size_t rowsAndCols) : Matrix(rowsAndCols, rowsAndCols) {}
  /**
   * @brief Copy Constructor for Matrix-object
   * @overload
   *
   * Makes a copy of an already existing Matrix-container with content.
   *
   * @param Matrix    A copy of a Matrix-class object.
   */
  Matrix(const Matrix &other)
      : Matrix(other.matrix_.size(),
               other.data_.size() / other.matrix_.size()) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = other.data_[i];
    }
  }
  /**
   * @brief Move Constructor for Matrix-object
   *
   * Moves a matrix.
   *
   * @param other     Matrix to be moved.
   */
  Matrix(Matrix &&other)
      : matrix_(std::move(other.matrix_)), data_(std::move(other.data_)) {
    for (size_t i = 0; i < matrix_.size(); i++) {
      matrix_[i] = data_.data() + (i * (data_.size() / matrix_.size()));
    }
  }
  /**
   * @brief T**() conversion overload.
   *
   * Useful for functions calls, e.g. fov_reverse().
   *
   */
  /* operator value_type **() { return matrix_.data(); } */
  operator std::span<value_type *>() {
    return std::span<value_type *>(matrix_);
  }
  // makes function call ambiguous for fov_reverse, since it takes two
  // matrices and double** and span are both valid.
  /**
   * @brief Enable row-access of Matrix container.
   * @overload []-operator
   *
   * Jump to virtual row of the Matrix.data_ std::vector.
   *
   * @param row       Row, that is supposed to be jumped to.
   */
  value_type *operator[](size_t row) {
    assert(row <= matrix_.size() - 1);
    return matrix_[row];
  }
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
  Matrix(std::vector<T *> &&rows, std::vector<T> &&data)
      : matrix_(std::move(rows)), data_(std::move(data)) {
    auto cols = data_.size() / matrix_.size();
    assert(cols * matrix_.size() == data_.size());
    for (size_t i = 0; i < matrix_.size(); i++) {
      matrix_[i] = data_.data() + (i * cols);
    }
  }
  /**
   * @brief Retrieving front of the matrix.
   *
   * Here for symmetric functionality to iterables like
   * std::vector.
   */
  value_type **data() { return matrix_.data(); }
  /**
   * @brief Retrieves dimensions of Matrix-object.
   */
  std::pair<size_t, size_t> shape() {
    return {matrix_.size(), data_.size() / matrix_.size()};
  }
  /**
   * @brief Fills entire Matrix Container with given value.
   *
   *@param filler     Value to fill the matrix with.
   */
  void fill(value_type filler = 0.0) {
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
ADOLC_API template <typename T = double> class Tensor {
  std::vector<T **> tensor_{};
  std::vector<T *> slices_{};
  std::vector<T> data_{};

public:
  using value_type = T;
  Tensor() = default;
  ~Tensor() = default;
  /**
   * @brief Move Constructor for Tensor-object
   *
   * Moves a matrix.
   *
   * @param other     Matrix to be moved.
   */
  Tensor(Tensor &&other)
      : tensor_(std::move(other.tensor_)), slices_(std::move(other.slices_)),
        data_(std::move(other.data_)) {
    size_t rows = tensor_.size();
    auto cols = slices_.size() / rows;
    auto dep = data_.size() / cols;
    for (size_t i = 0; i < rows * cols; i++) {
      slices_[i] = &data_[i * dep];
    }
    for (size_t i = 0; i < rows; i++) {
      tensor_[i] = &slices_[i * cols];
    }
  }
  Tensor &operator=(Tensor &&other) = default;
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
  Tensor(size_t rows, size_t cols, size_t dep, value_type filler = 0.0)
      : tensor_(rows), slices_(rows * cols), data_(rows * cols * dep, filler) {
    assert(rows > 0 && cols > 0 && dep > 0);
    for (size_t i = 0; i < rows * cols; i++) {
      slices_[i] = &data_[i * dep];
    }
    for (size_t i = 0; i < rows; i++) {
      tensor_[i] = &slices_[i * cols];
    }
  }
  /**
   * @brief Copy Constructor for Tensor-object
   * @overload
   *
   * Makes a copy of an already existing Tensor-container with content.
   *
   * @param Tensor    A copy of a Tensor-class object.
   */
  Tensor(const Tensor &other)
      : Tensor(other.tensor_.size(),
               other.slices_.size() / other.tensor_.size(),
               other.data_.size() / other.slices_.size()) {
    for (size_t i = 0; i < data_.size(); i++) {
      data_[i] = other.data_[i];
    }
  }
  /**
   * @brief T***() conversion overload.
   *
   * Useful for functions calls, e.g. fov_reverse().
   *
   */
  /* operator value_type ***() { return tensor_.data(); } */
  operator std::span<value_type **>() {
    return std::span<value_type **>(tensor_);
  }
  /**
   * @brief Enable row-access of Tensor container.
   * @overload []-operator
   *
   * Jump to virtual row of the Tensor.data_ array.
   *
   * @param row       Row, that is supposed to be jumped to.
   */
  value_type **operator[](size_t row) {
    assert(row <= tensor_.size() - 1);
    return tensor_[row];
  }
  /**
   * @brief Retrieving front of the Tensor.
   *
   * Here for symmetric functionality to iterables like
   * std::vector.
   */
  value_type ***data() { return tensor_.data(); }
  /**
   * @brief Fills entire Tensor Container with given value.
   *
   *@param filler     Value to fill the tensor with.
   */
  void fill(value_type filler) {
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
 * @overload Works also with std::array with derived notation.
 */
ADOLC_API template <
    typename T = double,
    template <typename, typename...> typename Cont = std::vector>
Cont<T> unitVector(size_t dim, size_t dir) {
  assert(dir <= dim);
  Cont<T> unitVec(dim, 0.0);
  unitVec[dir - 1] = static_cast<T>(1.0);
  return unitVec;
}
/**
 * @brief Unit Vector as std::array.
 * @overload Works also with std::vector-like containers.
 *
 * @tparam T   Data type.
 * @tparam dim Dimension of the vector.
 * @param dir      Unit  direction.
 * @return Unit vector as std::array.
 */
ADOLC_API template <typename T = double, size_t dim>
std::array<T, dim> unitVector(size_t dir) {
  assert(dir <= dim);
  std::array<T, dim> unitVec{};
  unitVec[dir - 1] = static_cast<T>(1.0);
  return unitVec;
}

namespace {
template <typename T> T fillIdentity(T identity) {
  for (size_t i = 0; i < identity.shape().first; i++) {
    identity[i][i] = static_cast<typename T::value_type>(1.0);
  }
  return identity;
}
} // namespace

/**
 * @brief Square Identity Matrix (Stack)
 *
 * Returns a Matrix-Container object, which is allocated
 * (using std::vector) on the stack.
 *
 * @tparam dim  Input/Output dimension.
 * @tparam T    data type.
 * @return Matrix of size Dim x Dim.
 */
ADOLC_API template <typename T = double, size_t dim = 1>
Matrix<T, dim, dim> unitMatrix() {
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
ADOLC_API template <typename T = double>
Matrix<T, 0, 0> unitMatrix(size_t dim) {
  auto identity = Matrix<T, 0, 0>{dim};
  return fillIdentity(identity);
}

#endif // _cplusplus

/****************************************************************************/
#endif // ADOLC_ADALLOC_H
