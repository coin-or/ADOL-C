#ifndef ADOLC_SPARSE_OPTIONS
#define ADOLC_SPARSE_OPTIONS

namespace ADOLC::Sparse {

/**
 * @brief Method used to determine the sparsity pattern.
 *
 * Selects the algorithm used to compute sparsity patterns of Jacobians
 * and Hessians.
 */
enum class SparseMethod {
  /**
   * @brief Index-domain propagation.
   *
   * Uses index-domain propagation as described in the ADOL-C documentation.
   * It is the default for sparsity detection.
   */
  IndexDomains,

  /**
   * @brief Bit-pattern propagation.
   *
   * Uses packed Boolean (bit-pattern) propagation through the computational
   * graph. This method enables strip-mining for large problems, at the cost of
   * additional memory.
   */
  BitPattern,
};

/**
 * @brief Control-flow handling mode for sparsity and dependence analysis.
 *
 * Determines how conditional branches in the taped computational graph
 * are treated during sparsity propagation.
 */
enum class ControlFlowMode {
  /**
   * @brief Safe (conservative) control-flow handling.
   *
   * All potential dependencies that may occur for any value of the independent
   * variables are included. No reevaluation at a basepoint is required.
   *
   * Example:
   * For `c = max(a, b)`, the dependence pattern of `c` is taken as the union
   * of the patterns of `a` and `b`.
   */
  Safe,

  /**
   * @brief Tight (basepoint-dependent) control-flow handling.
   *
   * Control-flow branches are resolved using the provided basepoint values.
   * The resulting sparsity pattern may contain fewer nonzeros but is only
   * valid locally around the basepoint where the function is analytic.
   *
   * Example:
   * For `c = max(a, b)`, the dependence pattern of `c` is taken from either
   * `a` or `b`, depending on which branch is active at the basepoint.
   */
  Tight,

  /**
   * @brief Legacy safe control-flow handling.
   *
   * Historical implementation of safe control-flow propagation retained
   * for backward compatibility.
   */
  OldSafe,

  /**
   * @brief Legacy tight control-flow handling.
   *
   * Historical implementation of tight control-flow propagation retained
   * for backward compatibility.
   */
  OldTight
};

/**
 * @brief Direction used for bit-pattern propagation.
 *
 * Specifies whether sparsity propagation is performed in forward mode,
 * reverse mode, or selected automatically based on problem dimensions.
 */
enum class BitPatternPropagationDirection {

  /**
   * @brief Automatic selection of propagation direction.
   *
   * A heuristic chooses forward or reverse propagation at runtime,
   * based on the relative sizes of the numbers of dependent
   * and independent variables.
   */
  Auto,

  /**
   * @brief Forward-mode bit-pattern propagation.
   *
   * Efficient when the number of independent variables is significantly
   * smaller than the number of dependent variables.
   */
  Forward,

  /**
   * @brief Reverse-mode bit-pattern propagation.
   *
   * Efficient when the number of dependent variables is significantly
   * smaller than the number of independent variables.
   */
  Reverse,
};

/**
 * @brief Compression orientation for sparse Jacobian recovery.
 *
 * Determines whether compressed derivative evaluations and recovery
 * are performed row-wise or column-wise.
 */
enum class CompressionMode {

  /**
   * @brief Column compression.
   *
   * Compresses columns of the Jacobian and recovers entries column-wise.
   * This is the default mode in ADOL-C.
   */
  Column,

  /**
   * @brief Row compression.
   *
   * Compresses rows of the Jacobian and recovers entries row-wise.
   */
  Row,
};

/**
 * @brief Recovery strategy for sparse Hessians.
 *
 * Selects the algorithm used by ColPack to recover the Hessian from
 * its compressed representation.
 */
enum class RecoveryMethod {

  /**
   * @brief Indirect (acyclic) recovery.
   *
   * Uses an acyclic coloring and recovers Hessian entries indirectly,
   * typically via successive substitutions.
   */
  Indirect,

  /**
   * @brief Direct (star) recovery.
   *
   * Uses star coloring and recovers Hessian entries directly without
   * additional arithmetic.
   */
  Direct,
};

} // namespace ADOLC::Sparse
#endif // ADOLC_SPARSE_OPTIONS