

#ifndef ADOLC_ABSNORMALFORM_CONCEPT_H
#define ADOLC_ABSNORMALFORM_CONCEPT_H
#include <concepts>
#include <type_traits>

namespace ADOLC {
/// @brief Base tag for abs-normal form shapes.
struct ANFShape {};

template <typename U>
concept DerivedFromANFShape =
    std::derived_from<std::remove_cvref_t<U>, ANFShape>;

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
  { t.dims() } -> DerivedFromANFShape;
  { t.resize(t.dims()) } -> std::same_as<void>;
  { t.updateCy() } -> std::same_as<void>;
  { t.updateCz() } -> std::same_as<void>;
};
} // namespace ADOLC
#endif // ADOLC_ABSNORMALFORM_CONCEPT_H