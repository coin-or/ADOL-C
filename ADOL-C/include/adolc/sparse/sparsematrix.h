#ifndef ADOLC_SPARSE_MATRIX_H
#define ADOLC_SPARSE_MATRIX_H

#include <adolc/drivers/absnormalformconcept.h>
#include <adolc/internal/common.h>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace ADOLC::Sparse {

/**
 * @brief One nonzero entry of a sparse matrix in coordinate format.
 *
 * The format stores the recovered nonzeros as triplets. This keeps row, column,
 * and value ownership tied together and makes the representation easier to move
 * and copy as one object.
 */
class CoordinateFormatTripled {
public:
  using coordinate_type = unsigned int;
  struct Coordinates {
    coordinate_type rowIndex_{0};
    coordinate_type colIndex_{0};
  };

private:
  Coordinates coords_;
  double value_{0.0};

public:
  CoordinateFormatTripled() = default;
  ~CoordinateFormatTripled() = default;

  CoordinateFormatTripled(const CoordinateFormatTripled &) = default;
  CoordinateFormatTripled &operator=(const CoordinateFormatTripled &) = default;

  CoordinateFormatTripled(CoordinateFormatTripled &&) noexcept = default;
  CoordinateFormatTripled &
  operator=(CoordinateFormatTripled &&) noexcept = default;

  /**
   * @brief Construct one coordinate-format triplet.
   *
   * @param coords Zero-based row index and column index
   * @param value  Numerical value stored at (`rowIdx`, `colIdx`).
   */
  constexpr CoordinateFormatTripled(Coordinates coords, double value)
      : coords_{coords}, value_(value) {}
  /**
   * @brief Construct one coordinate-format triplet.
   *
   * @param rowIndex Zero-based row index
   * @param colIndex Zero-based column index
   * @param value    Numerical value stored at (`rowIndex`, `colIndex`).
   */
  constexpr CoordinateFormatTripled(coordinate_type rowIndex,
                                    coordinate_type colIndex, double value)
      : coords_{Coordinates{rowIndex, colIndex}}, value_(value) {}

  /** @brief Get the row index. */
  coordinate_type rowIndex() const noexcept { return coords_.rowIndex_; }
  /** @brief Get the column index. */
  coordinate_type colIndex() const noexcept { return coords_.colIndex_; }
  /** @brief Get the numerical value. */
  double value() const noexcept { return value_; }
};

/**
 * @brief Owning sparse matrix container in coordinate-format.
 *
 * `SparseMatrix` stores the recovered nonzeros of a sparse matrix as a
 * contiguous array of `CoordinateFormatTripled` entries. The C++ sparse
 * drivers use this representation directly, while the legacy C drivers adapt
 * it back to separate `rind`, `cind`, and `values` arrays.
 */
class SparseMatrix {
public:
  using value_type = CoordinateFormatTripled;
  using storage_type = std::vector<value_type>;

private:
  storage_type entries_;

public:
  ~SparseMatrix() = default;
  /** @brief Construct an empty sparse matrix. */
  SparseMatrix() = default;
  /**
   * @brief Construct from already prepared coordinate-format entries.
   *
   * Ownership of the entry vector is transferred into the sparse matrix.
   *
   * @param entries Coordinate-format entries.
   */
  explicit SparseMatrix(storage_type &&entries) noexcept
      : entries_(std::move(entries)) {}

  /**
   * @brief Construct from an already prepared array of coordinate-format
   * entries.
   *
   * The entries of the array are copied.
   *
   * @param entries Array of size N with coordinate-format entries.
   */
  template <size_t N>
  explicit SparseMatrix(
      const std::array<CoordinateFormatTripled, N> &entries) noexcept
      : entries_(entries.begin(), entries.end()) {}
  /**
   * @brief Construct a sparse matrix with storage for `size` entries.
   *
   * Entries are value-initialized and may later be overwritten through
   * `operator[]`.
   *
   * @param size Number of coordinate-format entries to allocate.
   */
  explicit SparseMatrix(size_t size) : entries_(size) {}

  SparseMatrix(const SparseMatrix &other) = default;
  SparseMatrix &operator=(const SparseMatrix &other) = default;

  SparseMatrix(SparseMatrix &&other) noexcept = default;
  SparseMatrix &operator=(SparseMatrix &&other) noexcept = default;

  /** @brief Get mutable access to the underlying entry storage. */
  storage_type &entries() { return entries_; }
  /** @brief Get read-only access to the underlying entry storage. */
  const storage_type &entries() const { return entries_; }

  /** @brief Get the number of stored nonzeros. */
  size_t size() const { return entries_.size(); }
  /** @brief Check whether no entries are stored. */
  bool empty() const { return entries_.empty(); }

  /**
   * @brief Resize the number of stored entries.
   *
   * @param size New number of coordinate-format entries.
   */
  void resize(size_t size) { entries_.resize(size); }
  /** @brief Remove all entries while keeping vector ownership. */
  void clear() { entries_.clear(); }

  /**
   * @brief Append one entry at the end of the sparse matrix.
   *
   * @param value Entry to append.
   */
  void push_back(CoordinateFormatTripled value) { entries_.push_back(value); }

  /**
   * @brief Get the row index of one stored entry.
   *
   * @param idx Entry position.
   */
  unsigned int rowIndex(size_t idx) const { return entries_[idx].rowIndex(); }
  /**
   * @brief Get the column index of one stored entry.
   *
   * @param idx Entry position.
   */
  unsigned int colIndex(size_t idx) const { return entries_[idx].colIndex(); }
  /**
   * @brief Get the numerical value of one stored entry.
   *
   * @param idx Entry position.
   */
  double value(size_t idx) const { return entries_[idx].value(); }

  /**
   * @brief Mutable indexed access to one sparse entry.
   *
   * @param idx Entry position.
   */
  value_type &operator[](size_t idx) { return entries_[idx]; }
  /**
   * @brief Read-only indexed access to one sparse entry.
   *
   * @param idx Entry position.
   */
  const value_type &operator[](size_t idx) const { return entries_[idx]; }
};

namespace detail {

/** @brief Represents the different ANF blocks. */
enum class SparseANFBlock { Y, J, Z, L };

inline SparseANFBlock
classifySparseANFBlock(CoordinateFormatTripled::Coordinates coords, int depen,
                       int indep) {
  const bool isDependentRow =
      coords.rowIndex_ < static_cast<unsigned int>(depen);
  const bool isIndependentCol =
      coords.colIndex_ < static_cast<unsigned int>(indep);

  if (isDependentRow && isIndependentCol)
    return SparseANFBlock::Y;
  else if (isDependentRow)
    return SparseANFBlock::J;
  else if (isIndependentCol)
    return SparseANFBlock::Z;
  else
    return SparseANFBlock::L;
}
} // namespace detail

/** @brief Struct for storing the number of non-zero elements for each block. */
struct SparseShape : ANFShape {
  size_t y{0};
  size_t j{0};
  size_t z{0};
  size_t l{0};

  SparseShape() = default;
  SparseShape(size_t y, size_t j, size_t z, size_t l)
      : y(y), j(j), z(z), l(l) {};
};

/**
 * @brief Stores the abs-normal form in sparse format.
 *
 * This type is intended to satisfy the full `AbsNormalFormType` interface.
 *
 * The piecewise-linear sparse drivers recover the extended Jacobian
 * `[Y J; Z L]`. This type groups the four coordinate-format blocks in that
 * same order inside one owning object. Entries inside each block use
 * block-local coordinates, i.e. `Y` and `J` use dependent row indices, `Z` and
 * `L` use switching row indices, `Y` and `Z` use independent column indices,
 * and `J` and `L` use switching-variable column indices.
 *
 * The dense vectors `cy` and `cz` store the abs-normal constants associated
 * with `y` and `z`. They are computed by the sparse piecewise-linear driver
 * together with the four Jacobian blocks.
 */
struct SparseANF {
  using Shape = SparseShape;

  Shape shape{};

  /** @brief Output Jacobian block `Y = dy / dx`. */
  SparseMatrix Y;
  /** @brief Output block `J = dy / d|z|`. */
  SparseMatrix J;
  /** @brief Switching-equation Jacobian block `Z = dz / dx`. */
  SparseMatrix Z;
  /** @brief Switching-equation block `L = dz / d|z|`. */
  SparseMatrix L;

  /// Constant term of the switching equations.
  std::vector<double> cz;
  /// Constant term of the dependent equations.
  std::vector<double> cy;
  /// Dependent values at the evaluation point.
  std::vector<double> y;
  /// Switching-variable values at the evaluation point.
  std::vector<double> z;

  ~SparseANF() = default;
  SparseANF() = default;

  explicit SparseANF(Shape dims) { resize(dims); }

  SparseANF(SparseANF &&) = default;
  SparseANF &operator=(SparseANF &&) = default;

  SparseANF(const SparseANF &) = default;
  SparseANF &operator=(const SparseANF &) = default;

  /** @brief Check whether all sparse blocks and dense vectors are empty. */
  bool empty() const {
    return shape.j == 0 && shape.l == 0 && shape.y == 0 && shape.z == 0;
  }

  /** @brief Clear all sparse blocks and dense vectors. */
  void clear() {
    shape = {};
    Y.clear();
    J.clear();
    Z.clear();
    L.clear();
    cy.clear();
    cz.clear();
  }

  Shape dims() const { return shape; }

  /** @brief Compute `cy = y - J|z|` from the current data. */
  void updateCy() {
    cy = y;
    for (size_t i = 0; i < J.size(); ++i) {
      cy[J.rowIndex(i)] -= J.value(i) * std::fabs(z[J.colIndex(i)]);
    }
  }

  /** @brief Compute `cz = z - L|z|` from the current data. */
  void updateCz() {
    cz = z;
    for (size_t i = 0; i < L.size(); ++i) {
      cz[L.rowIndex(i)] -= L.value(i) * std::fabs(z[L.colIndex(i)]);
    }
  }

  /**
   * @brief Resize sparse blocks according to the given block counts.
   *
   * @param counts Number of entries in each block.
   */
  void resize(Shape dims) {
    shape = dims;
    Y.resize(shape.y);
    J.resize(shape.j);
    Z.resize(shape.z);
    L.resize(shape.l);
  }
};

} // namespace ADOLC::Sparse

#endif // ADOLC_SPARSE_MATRIX_H
