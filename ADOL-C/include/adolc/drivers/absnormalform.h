#ifndef ADOLC_ABSNORMALFORM_H
#define ADOLC_ABSNORMALFORM_H

#include <adolc/adolcexport.h>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace ADOLC {

/**
 * @brief Stores dimensions of an Abs-normal form.
 *
 * The shape records the sizes Abs-normal form blocks:
 * - `m`: number of dependent variables
 * - `n`: number of independent variables
 * - `s`: number of switching variables
 */
ADOLC_API struct ANFShape {
  size_t m{};
  size_t n{};
  size_t s{};
};

/**
 * @brief Structural concept for abs-normal form objects.
 *
 * A type models `AbsNormalFormType` when it exposes the six components of the
 * output-first abs-normal form
 * \f[
 *   \left[\begin{array}{c}
 *     y\\ z
 *   \end{array}\right]
 *   =
 *   \left[\begin{array}{c}
 *     cy\\ cz
 *   \end{array}\right]
 *   +
 *   \left[\begin{array}{cc}
 *     Y & J\\ Z & L
 *   \end{array}\right]
 *   \left[\begin{array}{c}
 *     x\\ |z|
 *   \end{array}\right].
 * \f]
 *
 * In addition to the eight components, the concept requires the
 * operations (`empty`, `clear`, `dims`, `resize`) and the two helper routines
 * that recompute the constant terms from the current matrices (`updateCy`,
 * `updateCz`).
 *
 * The concept does not prescribe a concrete storage layout, so both dense and
 * sparse representations may satisfy it as long as they provide this public
 * interface.
 */
template <typename T>
concept AbsNormalFormType = requires(T &t) {
  t.Y;
  t.J;
  t.Z;
  t.L;
  t.y;
  t.z;
  t.cy;
  t.cz;

  { t.empty() } -> std::convertible_to<bool>;
  { t.clear() } -> std::same_as<void>;
  { t.dims() } -> std::same_as<ANFShape>;
  { t.resize(t.dims()) } -> std::same_as<void>;
  { t.updateCy() } -> std::same_as<void>;
  { t.updateCz() } -> std::same_as<void>;
};

/**
 * @brief Abs-normal form container backed by contiguous storage.
 *
 * The flat `*_storage` vectors own the matrix entries, while `Y`, `J`, `Z`,
 * and `L` provide row-pointer views compatible with the existing driver
 * interfaces.
 */
ADOLC_API struct AbsNormalForm {
  using Shape = ANFShape;

  /// Logical dimensions of the represented ABS-normal form.
  Shape shape{};

  /// Row-pointer view of the `m x n` block `Y`.
  std::vector<double *> Y;
  /// Row-pointer view of the `m x s` block `J`.
  std::vector<double *> J;
  /// Row-pointer view of the `s x n` block `Z`.
  std::vector<double *> Z;
  /// Row-pointer view of the `s x s` block `L`.
  std::vector<double *> L;

  /// Owning contiguous storage for `Y`.
  std::vector<double> Y_storage;
  /// Owning contiguous storage for `J`.
  std::vector<double> J_storage;
  /// Owning contiguous storage for `Z`.
  std::vector<double> Z_storage;
  /// Owning contiguous storage for `L`.
  std::vector<double> L_storage;

  /// Constant term of the switching equations.
  std::vector<double> cz;
  /// Constant term of the dependent equations.
  std::vector<double> cy;
  /// Dependent values at the evaluation point.
  std::vector<double> y;
  /// Switching-variable values at the evaluation point.
  std::vector<double> z;

  ~AbsNormalForm() = default;
  AbsNormalForm() = default;

  /**
   * @brief Construct a dense ABS-normal form with the requested shape.
   *
   * Storage is allocated immediately using the same semantics as `resize()`.
   */
  explicit AbsNormalForm(Shape dims) { resize(dims); }

  AbsNormalForm(AbsNormalForm &&) = default;
  AbsNormalForm &operator=(AbsNormalForm &&) = default;

  AbsNormalForm(const AbsNormalForm &) = delete;
  AbsNormalForm &operator=(const AbsNormalForm &) = delete;

  /**
   * @brief Factory method that construct a Abs-normal form an existing tape.
   *
   * @param tapeId Tape identifier whose dependent, independent, and switching
   * dimensions determine the returned container size.
   */
  static AbsNormalForm fromTape(short tapeId);

  /**
   * @brief Report whether the shape is empty.
   *
   * This returns `true` exactly when all three dimensions are zero.
   */
  bool empty() const { return shape.m == 0 && shape.n == 0 && shape.s == 0; }

  /**
   * @brief Clear all stored data and reset the logical shape to `{0, 0, 0}`.
   *
   * All owning buffers and row-pointer views become empty.
   */
  void clear();

  /// @brief Return the logical Abs-normal form dimensions.
  Shape dims() const { return shape; }

  /** @brief Compute `cy = y - J|z|` from the current data. */
  void updateCy();

  /** @brief Compute `cz = z - L|z|` from the current data. */
  void updateCz();

  /**
   * @brief Resize the dense Abs-normal form based on given shape.
   *
   * All owning buffers are resized with vector semantics, and the row-pointer
   * views are rebuilt to point into the resized contiguous storage.
   */
  void resize(Shape dims);
};

} // namespace ADOLC

#endif // ADOLC_ABSNORMALFORM_H
