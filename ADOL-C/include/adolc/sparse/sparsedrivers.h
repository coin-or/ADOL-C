/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparsedrivers.h
 Revision: $Id$
 Contents: This file contains some "Easy To Use" interfaces of the sparse
package.

 Copyright (c) Andrea Walther

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#ifndef ADOLC_SPARSE_DRIVERS_H
#define ADOLC_SPARSE_DRIVERS_H
#include <adolc/adalloc.h>
#include <adolc/adolcerror.h>
#include <adolc/interfaces.h>
#include <adolc/internal/common.h>
#include <adolc/sparse/sparse_options.h>
#include <adolc/tape_interface.h>
#include <adolc/valuetape/sparseinfos.h>
#include <adolc/valuetape/valuetape.h>
#include <cstddef>
#include <span>
#include <vector>

// Max. number of unsigned ints to store the seed / jacobian matrix strips.
// Reduce this value to x if your system happens to run out of memory. x < 10
// makes no sense. x = 50 or 100 is better. x stays for (x * sizeof(size_t) * 8)
// (block) variables at once
#define PQ_STRIPMINE_MAX 30

namespace ADOLC::Sparse {
namespace detail {
// a word represents an instance of bitword_t every letter of the word gives a
// bit. This bit represents a depence of the input i (= location of the bit) to
// the output that corresponds to the word.
static constexpr size_t BITS_PER_WORD = 8 * sizeof(bitword_t);
static constexpr bitword_t MOST_SIGNIFICANT_BIT = static_cast<bitword_t>(1)
                                                  << (BITS_PER_WORD - 1);

/**
 * @brief Internal data structure managing buffers and parameters for bit-vector
 * propagation.
 *
 * This struct encapsulates the state and storage used during the computation
 * of Jacobian sparsity patterns using bit-vector propagation (BVP).
 *
 * It manages seed matrices, temporary buffers, and strip-mining batches,
 * depending on the chosen bit pattern propagation direction.
 *
 * @tparam BPPD
 *         Direction of bit pattern propagation — Forward or Reverse.
 */
template <BitPatternPropagationDirection BPPD> struct BvpData {
  int rc_{0};            ///< Return code from ADOL-C drivers.
  int depen_{-1};        ///< Number of dependent (output) variables.
  int indep_{-1};        ///< Number of independent (input) variables.
  int wordsPerBatch_{0}; ///< Number of bit words per batch.
  int bitsPerStrip_{0};  ///< Number of bits represented by one batch.
  int numStripmineBatches_{
      0}; ///< Number of batches needed to cover all variables.
  std::vector<unsigned char>
      indepWordHasNonzero_; ///< Flags marking independent variables that appear
                            ///< in nonzero entries.
  std::vector<double> valuepoint_; ///< Evaluation point values (used in tight
                                   ///< control-flow mode).
  std::vector<bitword_t *> seed_;  ///< Seed matrix partitions.
  std::vector<bitword_t *>
      jacobianBitPattern_; ///< Jacobian bit pattern partitions.

  BvpData(const BvpData &) = delete;
  BvpData &operator=(const BvpData &) = delete;
  BvpData(BvpData &&) = delete;
  BvpData &operator=(BvpData &&) = delete;

  /**
   * @brief Destructor — frees dynamically allocated memory for bit patterns.
   */
  ~BvpData() {
    for (auto &s : seed_)
      delete[] s;
    for (auto &j : jacobianBitPattern_)
      delete[] j;
  }

  /**
   * @brief Constructs a bit-vector propagation data container.
   *
   * Initializes memory and parameters for strip-mining based on the
   * propagation direction and number of variables.
   *
   * @param depen Number of dependent variables.
   * @param indep Number of independent variables.
   */
  BvpData(int depen, int indep) : depen_(depen), indep_(indep) {
    int numWordsFullSeed = 0;
    if constexpr (BPPD == BitPatternPropagationDirection::Forward)
      numWordsFullSeed =
          indep_ / BITS_PER_WORD + ((indep_ % BITS_PER_WORD) != 0);
    else if (BPPD == BitPatternPropagationDirection::Reverse)
      numWordsFullSeed = depen / BITS_PER_WORD + ((depen % BITS_PER_WORD) != 0);

    wordsPerBatch_ = (numWordsFullSeed <= PQ_STRIPMINE_MAX) ? numWordsFullSeed
                                                            : PQ_STRIPMINE_MAX;
    bitsPerStrip_ = wordsPerBatch_ * BITS_PER_WORD;
    numStripmineBatches_ =
        (numWordsFullSeed <= PQ_STRIPMINE_MAX)
            ? 1
            : numWordsFullSeed / PQ_STRIPMINE_MAX +
                  ((numWordsFullSeed % PQ_STRIPMINE_MAX) != 0);

    if constexpr (BPPD == BitPatternPropagationDirection::Forward) {
      seed_.resize(indep_);
      for (auto &s : seed_)
        s = new bitword_t[wordsPerBatch_];

      jacobianBitPattern_.resize(depen_);
      for (auto &j : jacobianBitPattern_)
        j = new bitword_t[wordsPerBatch_];

      indepWordHasNonzero_.resize(bitsPerStrip_);

    } else if (BPPD == BitPatternPropagationDirection::Reverse) {
      seed_.resize(wordsPerBatch_);
      for (auto &s : seed_)
        s = new bitword_t[depen_];

      jacobianBitPattern_.resize(wordsPerBatch_);
      for (auto &j : jacobianBitPattern_)
        j = new bitword_t[indep_];

      indepWordHasNonzero_.resize(indep_);
    }
  }
};

/**
 * @brief Verifies input data consistency for bit-vector propagation.
 *
 * @tparam CFM Control flow mode (Safe, Tight, etc.).
 * @param basepoint Pointer to the independent variable values (may be null in
 * Safe mode).
 */
template <ControlFlowMode CFM> void checkBVPInput(const double *) {};

/**
 * @brief Resets all entries in the compressed row storage pointer array to
 * nullptr.
 *
 * @param compressedRowStorage Span over the row storage pointer array to be
 * reset.
 */
inline void resetInput(std::span<uint *> &compressedRowStorage) {
  for (auto &row : compressedRowStorage) {
    row = nullptr;
  }
}

/**
 * @brief Sets all seed matrix entries to zero for reuse in the next strip-mined
 * batch.
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @param data  Bit-vector propagation state data.
 * @param dim2  Inner dimension to clear (depends on direction).
 */
template <BitPatternPropagationDirection BPPD>
void resetOldSeed(BvpData<BPPD> &data, size_t dim2) {
  for (auto &s : data.seed_)
    for (size_t j = 0; j < dim2; ++j)
      s[j] = 0;
}

/**
 * @brief Builds a strip-mined partition of the seed matrix for a given batch.
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @param indep   Number of independent variables.
 * @param stripIdx Current strip-mined batch index.
 * @param data     Bit-vector propagation state data.
 */
template <BitPatternPropagationDirection BPPD>
void prepareSeedMatrix(size_t stripIdx, BvpData<BPPD> &data) {
  if constexpr (BPPD == BitPatternPropagationDirection::Forward) {
    resetOldSeed(data, data.wordsPerBatch_);
    createSeed(data, stripIdx, data.indep_);
  } else if (BPPD == BitPatternPropagationDirection::Reverse) {
    resetOldSeed(data, data.depen_);
    createSeed(data, stripIdx, data.depen_);
  }
}

/**
 * @brief Sets a single bit in the seed matrix corresponding to a specific
 * variable index.
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @param data                Bit-vector propagation state data.
 * @param idx                 Global variable index.
 * @param currentStripWordIdx Starting index of the current strip.
 */
template <BitPatternPropagationDirection BPPD>
void setSeed(BvpData<BPPD> &data, size_t idx, size_t currentStripWordIdx) {
  if constexpr (BPPD == BitPatternPropagationDirection::Forward)
    data.seed_[idx][(idx - currentStripWordIdx) / BITS_PER_WORD] =
        MOST_SIGNIFICANT_BIT >> ((idx - currentStripWordIdx) % BITS_PER_WORD);

  else if (BPPD == BitPatternPropagationDirection::Reverse)
    data.seed_[(idx - currentStripWordIdx) / BITS_PER_WORD][idx] =
        MOST_SIGNIFICANT_BIT >> ((idx - currentStripWordIdx) % BITS_PER_WORD);
}

/**
 * @brief Creates a seed matrix segment by invoking setSeed() for each variable
 * in the strip.
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @param data  Bit-vector propagation data container.
 * @param stripIdx Index of the strip-mined batch.
 * @param end   End index (exclusive) of the variable range for this batch.
 */
template <BitPatternPropagationDirection BPPD>
void createSeed(BvpData<BPPD> &data, size_t stripIdx, size_t end) {
  size_t currentStripWordIdx = stripIdx * data.bitsPerStrip_;
  size_t nextStripWordIdx = (stripIdx + 1) * data.bitsPerStrip_;
  if (nextStripWordIdx > end)
    nextStripWordIdx = end;

  for (size_t idx = currentStripWordIdx; idx < nextStripWordIdx; idx++)
    setSeed(data, idx, currentStripWordIdx);
}

/**
 * @brief Extracts the sparsity pattern for one dependent variable (row) in
 * forward bit-vector propagation mode.
 *
 * This routine constructs or extends a row in compressed row storage format
 * based on the current bit pattern flags in @p indepWordHasNonzero_.
 *
 * Each bit set in the propagated Jacobian pattern indicates a dependency of
 * the current dependent variable on an independent variable.
 *
 * @param wordIdx Index of the dependent variable (row in the Jacobian).
 * @param stripIdx Index of the current strip-mined batch.
 * @param currentStripWordIdx Starting global index of the strip.
 * @param data Bit-vector propagation data container (Forward mode).
 * @param[in,out] compressedRowStorage
 *        Span of pointers to CRS rows (updated in-place).
 *
 * @note
 *  - The function merges new nonzeros with previously existing entries
 *    when processing subsequent strips.
 *  - Memory for each row is reallocated as needed and must be freed by the
 * caller.
 */
void extract(size_t wordIdx, size_t stripIdx, size_t currentStripWordIdx,
             BvpData<BitPatternPropagationDirection::Forward> &data,
             std::span<uint *> &compressedRowStorage);

/**
 * @brief Extracts the sparsity pattern for one dependent variable (row)
 * in reverse bit-vector propagation mode.
 *
 * This function builds the corresponding CRS row for
 * each output, using the flags stored in @p indepWordHasNonzero_.
 *
 * Each bit set in the propagated Jacobian pattern indicates a dependency of
 * the current dependent variable on an independent variable.

 * @param wordIdx Index of the dependent variable (row in the Jacobian).
 * @param stripIdx Index of the current strip-mined batch.
 * @param data Bit-vector propagation data container (Reverse mode).
 * @param[in,out] compressedRowStorage
 *        Span of pointers to CRS rows (updated in-place).
 *
 * @note
 *  - Memory for each row is reallocated every time the strip is processed.
 *  - The caller is responsible for freeing the allocated CRS memory.
 */
void extract(size_t wordIdx, size_t stripIdx,
             BvpData<BitPatternPropagationDirection::Reverse> &data,
             std::span<uint *> &compressedRowStorage);

/**
 * @brief Converts Jacobian bit patterns to compressed row storage (CRS) format
 * for one strip.
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @param stripIdx             Index of the strip-mined batch.
 * @param compressedRowStorage Output CRS array (span over row pointers).
 * @param data                 Bit-vector propagation data container.
 */
template <BitPatternPropagationDirection BPPD>
void extractCompressedRowStorage(int stripIdx,
                                 std::span<uint *> &compressedRowStorage,
                                 BvpData<BPPD> &data) {
  int currentStripWordIdx = stripIdx * data.bitsPerStrip_;
  int nextStripWordIdx = (stripIdx + 1) * data.bitsPerStrip_;
  if constexpr (BPPD == BitPatternPropagationDirection::Forward)
    for (int wordIdx = 0; wordIdx < data.depen_; wordIdx++) {
      int idx = 0;
      bitword_t currentBit = MOST_SIGNIFICANT_BIT;
      for (int i = 0; i < data.bitsPerStrip_; ++i) {
        if (currentBit == 0) {
          currentBit = MOST_SIGNIFICANT_BIT;
          idx++;
        }
        if (currentBit & data.jacobianBitPattern_[wordIdx][idx])
          data.indepWordHasNonzero_[i] = 1;

        currentBit = currentBit >> 1;
      }
      extract(wordIdx, stripIdx, currentStripWordIdx, data,
              compressedRowStorage);
    }
  else if (BPPD == BitPatternPropagationDirection::Reverse) {
    if (nextStripWordIdx > data.depen_)
      nextStripWordIdx = data.depen_;
    int idx = 0;
    bitword_t currentBit = MOST_SIGNIFICANT_BIT;
    for (int wordIdx = currentStripWordIdx; wordIdx < nextStripWordIdx;
         wordIdx++) {
      if (currentBit == 0) {
        currentBit = MOST_SIGNIFICANT_BIT;
        idx++;
      }
      for (int i = 0; i < data.indep_; i++) {
        if (currentBit & data.jacobianBitPattern_[idx][i]) {
          data.indepWordHasNonzero_[i] = 1;
        }
      }
      currentBit = currentBit >> 1;
      extract(wordIdx, stripIdx, data, compressedRowStorage);
    }
  }
}

/**
 * @brief Prepares reverse call using zos_forward.
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @tparam CFM  Control flow mode.
 * @param tapeId    ADOL-C tape identifier.
 * @param basepoint Pointer to basepoint values (used in tight control-flow
 * mode).
 * @param data      Bit-vector propagation data container.
 */
template <BitPatternPropagationDirection BPPD, ControlFlowMode CFM>
void prepareADCalls(short tapeId, const double *basepoint,
                    BvpData<BPPD> &data) {
  if constexpr (BPPD == BitPatternPropagationDirection::Reverse &&
                CFM == ControlFlowMode::Tight)
    data.rc_ = zos_forward(tapeId, data.depen_, data.indep_, 1, basepoint,
                           data.valuepoint_.data());
}

/**
 * @brief Executes the ADOL-C internal bit-vector propagation calls (forward or
 * reverse).
 *
 * @tparam BPPD Bit pattern propagation direction.
 * @tparam CFM  Control flow mode.
 * @param tapeId    ADOL-C tape identifier.
 * @param basepoint Pointer to independent variable values (required for forward
 * + tight).
 * @param data      Bit-vector propagation state container.
 */
template <BitPatternPropagationDirection BPPD, ControlFlowMode CFM>
void ADCalls(short tapeId, const double *basepoint, BvpData<BPPD> &data) {
  if constexpr (BPPD == BitPatternPropagationDirection::Forward) {
    if constexpr (CFM == ControlFlowMode::Tight) {
      data.valuepoint_.resize(data.depen_);
      data.rc_ = int_forward_tight(tapeId, data.depen_, data.indep_,
                                   data.wordsPerBatch_, basepoint,
                                   data.seed_.data(), data.valuepoint_.data(),
                                   data.jacobianBitPattern_.data());
    } else if (CFM == ControlFlowMode::Safe)
      data.rc_ = int_forward_safe(tapeId, data.depen_, data.indep_,
                                  data.wordsPerBatch_, data.seed_.data(),
                                  data.jacobianBitPattern_.data());

  } else if (BPPD == BitPatternPropagationDirection::Reverse) {
    if constexpr (CFM == ControlFlowMode::Tight)
      data.rc_ = int_reverse_tight(tapeId, data.depen_, data.indep_,
                                   data.wordsPerBatch_, data.seed_.data(),
                                   data.jacobianBitPattern_.data());
    else if (CFM == ControlFlowMode::Safe)
      data.rc_ = int_reverse_safe(tapeId, data.depen_, data.indep_,
                                  data.wordsPerBatch_, data.seed_.data(),
                                  data.jacobianBitPattern_.data());
  }
}

/**
 * @brief Main internal implementation for bit-vector propagation.
 *
 * Iteratively builds and propagates seed matrices, performs ADOL-C bit-vector
 * propagation calls, and extracts the sparsity pattern into CRS format.
 *
 * @tparam CFM  Control flow mode.
 * @tparam BPPD Bit pattern propagation direction.
 * @param tapeId    ADOL-C tape identifier.
 * @param depen     Number of dependent variables.
 * @param indep     Number of independent variables.
 * @param basepoint Pointer to basepoint values.
 * @param compressedRowStorage Output CRS sparsity pattern.
 * @return Return code from ADOL-C driver.
 */
template <ControlFlowMode CFM, BitPatternPropagationDirection BPPD>
int bitVectorPropagation(short tapeId, int depen, int indep,
                         const double *basepoint,
                         std::span<uint *> &compressedRowStorage) {
  using namespace detail;
  checkBVPInput<CFM>(basepoint);
  BvpData<BPPD> data{depen, indep};
  prepareADCalls<BPPD, CFM>(tapeId, basepoint, data);
  for (int stripIdx = 0; stripIdx < data.numStripmineBatches_; stripIdx++) {
    prepareSeedMatrix(stripIdx, data);
    ADCalls<BPPD, CFM>(tapeId, basepoint, data);
    extractCompressedRowStorage(stripIdx, compressedRowStorage, data);
  }
  return data.rc_;
}

} // namespace detail

/**
 * @brief Computes the sparsity pattern of a Jacobian using bit-vector
 * propagation.
 *
 * This routine determines the sparsity pattern of the Jacobian of a taped
 * function by propagating bit patterns through the recorded computational
 * graph. It supports both forward and reverse propagation strategies, with
 * optional control-flow sensitivity.
 *
 * @param tag
 *        Tape identification number for the recorded function.
 * @param depen
 *        Number of dependent (output) variables.
 * @param indep
 *        Number of independent (input) variables.
 * @param basepoint
 *        Pointer to an array of length `indep` specifying the values of the
 *        independent variables at which the sparsity pattern is evaluated.
 *        Required if `options.cfmode_ == ControlFlowMode::Tight`, otherwise
 *        may be `nullptr`.
 * @param compressedRowStorage
 *        Output: Compressed Row Storage (compressedRowStorage) representation
 * of the Jacobian sparsity pattern.
 *        - `compressedRowStorage[i][0]` → number of nonzero entries in row
 * `i`
 *        - `compressedRowStorage[i][1..]` → column indices (0-based) of
 * nonzero entries in row `i`. Memory for `compressedRowStorage[i]` is
 * (re)allocated within this routine.
 * @param options
 *        DriverOptions controlling propagation and control-flow handling:
 *        - `options.bpdir_` (BitPatternPropagationDirection):
 *            - `Automatic` → heuristic selection (forward if `depen >=
 * indep/2`, else reverse)
 *            - `Forward`   → forward mode propagation
 *            - `Reverse`   → reverse mode propagation
 *        - `options.cfmode_` (ControlFlowMode):
 *            - `Safe`  → control flow branches may be ignored
 *            - `Tight` → control flow branches are tested at `basepoint`
 *
 * @return
 *        Return code from the underlying forward/reverse driver:
 *        - `0` → success, no warnings
 *        - `1` → warnings occurred
 *        - `2` → error occurred during propagation
 *        - `3` → default initialization value (no propagation done)
 *
 * @note
 *  - Uses "strip-mining" to split the bit pattern matrix into manageable
 * blocks if the number of bits per `size_t` is insufficient to store the full
 * pattern.
 *  - In forward mode, seeds are built per independent variable block;
 *    in reverse mode, seeds are built per dependent variable block.
 *  - Memory allocated for `compressedRowStorage[i]` must be freed by the
 * caller when no longer needed.
 *
 * @see jac_pat
 * @see hess_pat
 */
template <ControlFlowMode CFM, BitPatternPropagationDirection BPPD =
                                   BitPatternPropagationDirection::Auto>
ADOLC_API int bit_vector_propagation(short tapeId, int depen, int indep,
                                     const double *basepoint,
                                     std::span<uint *> &compressedRowStorage) {
  using namespace detail;
  if constexpr (BPPD == BitPatternPropagationDirection::Auto) {
    // we allow here run-time selection
    if (depen >= indep / 2)
      return bitVectorPropagation<CFM, BitPatternPropagationDirection::Forward>(
          tapeId, depen, indep, basepoint, compressedRowStorage);

    else
      return bitVectorPropagation<CFM, BitPatternPropagationDirection::Reverse>(
          tapeId, depen, indep, basepoint, compressedRowStorage);
  } else
    return bitVectorPropagation<CFM, BPPD>(tapeId, depen, indep, basepoint,
                                           compressedRowStorage);
}

/**
 * @brief Compute Jacobian sparsity pattern using the selected sparse method.
 *
 * Dispatches to either index-domain propagation or bit-vector propagation
 * depending on the template parameter @p SM and control-flow mode @p CFM.
 *
 * @tparam SM   SparseMethod used to compute pattern:
 *              - SparseMethod::IndexDomains : index-domain propagation
 *              - SparseMethod::BitPattern   : bit-vector propagation
 * @tparam CFM  ControlFlowMode controlling control-flow sensitivity:
 *              - ControlFlowMode::Safe  : conservative (may ignore branches)
 *              - ControlFlowMode::Tight : evaluate control-flow at @p basepoint
 * @tparam BPPD BitPatternPropagationDirection used when SM==BitPattern.
 *             If BitPatternPropagationDirection::Auto a runtime heuristic
 *             selects forward/reverse direction.
 *
 * @param tag                   Tape identifier (as used by ADOL-C tracing).
 * @param depen                 Number of dependent (output) variables (rows).
 * @param indep                 Number of independent (input) variables (cols).
 * @param basepoint             Pointer to an array of length @p indep. Required
 *                              when @p CFM == ControlFlowMode::Tight; may be
 *                              nullptr in Safe mode.
 * @param[out] compressedRowStorage
 *                              Span of length @p depen where each element is a
 *                              pointer to a Compressed Row Storage (CRS) row:
 *                              - entry [i][0] holds the count of nonzeros in
 *                                row i
 *                              - entry [i][1..] holds the 0-based column
 *                                indices that represent the influencing
 *                                independs
 *
 * @return  Non-negative: number of nonzeros found; Negative: error code from
 * the underlying propagation routine.
 *
 * @note Ownership: allocated row buffers in compressedRowStorage are
 * transferred to the caller / tape infrastructure and must be freed
 * appropriately.
 */
template <
    SparseMethod SM, ControlFlowMode CFM,
    BitPatternPropagationDirection BPPD = BitPatternPropagationDirection::Auto>
ADOLC_API int jac_pat(short tag, int depen, int indep, const double *basepoint,
                      std::span<uint *> &compressedRowStorage) {
  detail::resetInput(compressedRowStorage);
  if constexpr (SM == SparseMethod::IndexDomains &&
                CFM == ControlFlowMode::Tight)
    return indopro_forward_tight(tag, depen, indep, basepoint,
                                 compressedRowStorage.data());
  else if (SM == SparseMethod::IndexDomains && CFM == ControlFlowMode::Safe)
    return indopro_forward_safe(tag, depen, indep, basepoint,
                                compressedRowStorage.data());
  else if (SM == SparseMethod::BitPattern)
    return bit_vector_propagation<CFM, BPPD>(tag, depen, indep, basepoint,
                                             compressedRowStorage);
};

/**
 * @brief Generate a seed matrix (coloring) for compressed Jacobian recovery.
 *
 * Uses ColPack's Bipartite graph coloring to produce the seed matrix suitable
 * for recovering the full Jacobian from compressed directional evaluations.
 * This function uses ColPack's unmanaged API and returns pointers via @p Seed
 * and the compressed dimension via @p p.
 *
 * @tparam CM CompressionMode selecting orientation of compression:
 *            - CompressionMode::Row    : seeds correspond to row-compression
 *            - CompressionMode::Column : seeds correspond to column-compression
 *
 * @param m     Number of dependent variables (rows of JP).
 * @param n     Number of independent variables (columns of JP).
 * @param JP    Compressed Row Storage, storing the dependencies on
 *              independents.
 * @param[out] Seed  Output pointer to a 2D array representing the seed matrix.
 *                   Memory is provided by ColPack (unmanaged) and returned to
 *                   caller.
 * @param[out] p     On return: compressed dimension (number of seed columns
 *                   when Row compression, or seed rows when Column
 *                   compression).
 *
 * @note The caller is responsible for managing the returned @p Seed memory
 *       according to ColPack's unmanaged API semantics.
 */
template <CompressionMode CM>
ADOLC_API void generate_seed_jac(int m, int n, const std::span<uint *> JP,
                                 double ***Seed, int *p) {
  generateSeedJac<CM>(m, n, JP, Seed, p);
}

namespace detail {

/**
 * @brief Allocates and computes pattern of jacobian and computes a seed to
 * compute a sparse Jacobian.
 *
 * This routine computes the Jacobian sparsity pattern (via `jac_pat`) and then
 * uses ColPack to generate a seed matrix (graph coloring) suitable for
 * compressed Jacobian recovery. It stores results inside the tape's
 * `sJInfos()` structure so subsequent `compute_sparse_jac`/repeat calls can
 * reuse the seed and recovery objects.
 *
 * @tparam SM
 *         Sparsity-detection method used by `jac_pat`:
 *         - `SparseMethod::IndexDomains` : index domain propagation
 *         - `SparseMethod::BitPattern`  : bit-vector propagation
 * @tparam CM
 *         Compression orientation for ColPack seed generation:
 *         - `CompressionMode::Row`    : row-compression (recover by rows)
 *         - `CompressionMode::Column` : column-compression (recover by columns)
 * @tparam CFM
 *         Control-flow mode used when computing the sparsity pattern.
 * @tparam BPPD
 *         Bit pattern propagation direction when `SM == BitPattern`. When
 *         `BitPatternPropagationDirection::Auto` a runtime heuristic is used.
 *
 * @param tag        ADOL-C tape identifier.
 * @param depen      Number of dependent (output) variables (rows of Jacobian).
 * @param indep      Number of independent (input) variables (columns of
 * Jacobian).
 * @param basepoint  Pointer to an array of length `indep` with the basepoint
 *                   values. May be `nullptr` if control-flow mode does not
 *                   require a basepoint.
 * @param[out] nnz   Pointer to an integer where the computed number of
 *                   nonzeros in the Jacobian pattern will be stored.
 *
 * @return
 *   - >= 0 : Number of seed columns (when CM==Row) or seed rows (when
 * CM==Column) produced by ColPack (i.e., the compressed dimension).
 *   - <  0 : Error code forwarded from `jac_pat` or other internal failures.
 *
 * @note
 *  - The function allocates and stores the Compressed Row Storage (CRS)
 *    pattern inside `tape.sJInfos().JP_`. Ownership of those row buffers is
 *    transferred to the tape (they will be freed/managed by the tape
 * lifecycle).
 *  - After success, `tape.sJInfos().g_` and `tape.sJInfos().jr1d_` are created
 *    and kept in the tape for later use by `compute_sparse_jac`.
 *  - `*nnz` is set to the number of nonzero entries found in the Jacobian.
 */
template <
    SparseMethod SM, CompressionMode CM, ControlFlowMode CFM,
    BitPatternPropagationDirection BPPD = BitPatternPropagationDirection::Auto>
int buildJacPatternAndSeed(short tag, int depen, int indep,
                           const double *basepoint, int *nnz) {

  ValueTape &tape = findTape(tag);
  tape.sJInfos().setJP(std::vector<uint *>(depen));
  std::span<uint *> JPSpan = tape.sJInfos().getJP();
  int ret_val = jac_pat<SM, CFM, BPPD>(tag, depen, indep, basepoint, JPSpan);

  if (ret_val < 0) {
    printf(" ADOL-C error in sparse_jac() \n");
    return ret_val;
  }

  tape.sJInfos().depen_ = depen;
  tape.sJInfos().nnzIn_ = 0;
  for (int i = 0; i < depen; i++) {
    for (uint j = 1; j <= tape.sJInfos().JP_[i][0]; j++)
      tape.sJInfos().nnzIn_++;
  }

  *nnz = tape.sJInfos().nnzIn_;
  tape.sJInfos().initColoring(depen, indep);

  if constexpr (CM == CompressionMode::Row) {
    tape.sJInfos().generateSeedJac("ROW_PARTIAL_DISTANCE_TWO");
    tape.sJInfos().seedClms_ = indep;
    ret_val = tape.sJInfos().seedRows_;
  } else if (CM == CompressionMode::Column) {
    tape.sJInfos().generateSeedJac("COLUMN_PARTIAL_DISTANCE_TWO");
    tape.sJInfos().seedRows_ = depen;
    ret_val = tape.sJInfos().seedClms_;
  }
  return ret_val;
}

/**
 * @brief Compute sparse Jacobian using precomputed seed matrix and recovery
 * information.
 *
 * This routine performs the actual compressed Jacobian computation using the
 * seed matrix and graph/recovery structures stored in the tape's `sJInfos()`.
 * It supports both row- and column-oriented compression strategies and both
 * user-allocated and internally allocated (unmanaged) output buffers.
 *
 * @tparam CM
 *        Compression orientation (Row or Column) chosen when the seed was
 *        generated.
 *
 * @param tag        ADOL-C tape identifier.
 * @param depen      Number of dependent variables (rows).
 * @param indep      Number of independent variables (columns).
 * @param basepoint  Pointer to basepoint values (length `indep`) used for
 *                   evaluation in tight control-flow or other drivers.
 * @param nnz        Expected number of nonzeros (must match
 *                   `tape.sJInfos().nnzIn_`). On exit: unchanged.
 * @param[out] rind  Pointer to pointer which will receive the row-index array
 *                   of the coordinate-format Jacobian (allocated by this
 *                   function if `*rind == nullptr`, otherwise user-provided).
 * @param[out] cind  Pointer to pointer which will receive the column-index
 *                   array of the coordinate-format Jacobian (allocated by
 *                   this function if `*cind == nullptr`, otherwise
 * user-provided).
 * @param[out] values Pointer to pointer which will receive the nonzero values
 *                    (allocated by this function if `*values == nullptr`,
 *                    otherwise user-provided).
 *
 * @return
 *   - `0` on success (or driver-specific non-negative codes),
 *   - negative error code on failure (propagated from internal ADOL-C calls).
 *
 * @pre
 *  - `buildJacPatternAndSeed` must have been called previously for this tape to
 *    populate `tape.sJInfos()` with a valid seed (`Seed_`), graph (`g_`) and
 *    recovery object (`jr1d_`).
 *  - `*nnz` must equal the number of nonzeros discovered earlier and stored
 *    in `tape.sJInfos().nnzIn_`. If not, the function returns error `-3`.
 *
 * @post
 *  - On success, `rind`, `cind`, and `values` point to coordinate-format
 *    arrays describing the sparse Jacobian. If the user supplied non-null
 *    pointers, user memory is used; otherwise memory is allocated by the
 *    recovery routine and must be freed by the caller.
 *
 * @note
 *  - The function allocates intermediate storage `B_` and `y_` inside the
 *    tape; these are freed with the tape lifecycle or overwritten on subsequent
 * calls.
 *  - For `CompressionMode::Row`, a forward evaluation `zos_forward` is used to
 *    compute `y_` followed by `fov_reverse` to obtain `B_`. For
 *    `CompressionMode::Column`, `fov_forward` is used.
 *  - The ColPack `RecoverD2*` functions are invoked to transform the compressed
 *    representation to coordinate format; they accept both user-managed and
 *    unmanaged memory usage patterns (usermem vs unmanaged).
 */
template <CompressionMode CM>
int computeSparseJac(short tag, int depen, int indep, const double *basepoint,
                     int *nnz, unsigned int **rind, unsigned int **cind,
                     double **values) {
  ValueTape &tape = findTape(tag);
  int ret_val = 0;
  myfree2(tape.sJInfos().B_);
  myfree1(tape.sJInfos().y_);
  tape.sJInfos().B_ =
      myalloc2(tape.sJInfos().seedRows_, tape.sJInfos().seedClms_);
  tape.sJInfos().y_ = myalloc1(depen);

  if (tape.sJInfos().nnzIn_ != *nnz) {
    printf(" ADOL-C error in sparse_jac():"
           " Number of nonzeros not consistent,"
           " repeat call with repeat = 0 \n");
    return -3;
  }

  if constexpr (CM == CompressionMode::Row) {
    ret_val = zos_forward(tag, depen, indep, 1, basepoint, tape.sJInfos().y_);
    if (ret_val < 0)
      return ret_val;
    MINDEC(ret_val, fov_reverse(tag, depen, indep, tape.sJInfos().seedRows_,
                                tape.sJInfos().Seed_, tape.sJInfos().B_));
  } else if (CM == CompressionMode::Column)
    ret_val =
        fov_forward(tag, depen, indep, tape.sJInfos().seedClms_, basepoint,
                    tape.sJInfos().Seed_, tape.sJInfos().y_, tape.sJInfos().B_);

  if (values != nullptr && *values != nullptr && rind != nullptr &&
      *rind != nullptr && cind != nullptr && *cind != nullptr) {
    // everything is preallocated, we assume correctly
    // call usermem versions
    if (CM == CompressionMode::Row)
      tape.sJInfos().recoverRowFormatUserMem(rind, cind, values);
    else if (CM == CompressionMode::Column)
      tape.sJInfos().recoverColFormatUserMem(rind, cind, values);
  } else {
    // at least one of rind cind values is not allocated, deallocate others
    // and call unmanaged versions
    if (values != nullptr && *values != nullptr)
      free(*values);
    if (rind != nullptr && *rind != nullptr)
      free(*rind);
    if (cind != nullptr && *cind != nullptr)
      free(*cind);
    if (CM == CompressionMode::Row) {
      tape.sJInfos().recoverRowFormat(rind, cind, values);
    } else if (CM == CompressionMode::Column) {
      tape.sJInfos().recoverColFormat(rind, cind, values);
    }
  }
  return ret_val;
}

} // namespace detail

/**
 * @brief High-level API to compute a sparse Jacobian (single call or repeat).
 *
 * This function coordinates sparsity detection, seed generation (via ColPack),
 * compressed AD evaluations, and recovery. When @p repeat == 0 the function
 * computes and caches the sparsity/seed information (stored on the tape).
 * Subsequent calls (repeat != 0) reuse cached seeds to compute numerical
 * Jacobian values more efficiently.
 *
 * @tparam SM   Default SparseMethod used for pattern detection (IndexDomains).
 * @tparam CM   Default CompressionMode used for recovery (Column).
 * @tparam CFM  Default ControlFlowMode for propagation (Safe).
 * @tparam BPPD Default bit-propagation direction (Auto).
 *
 * @param tag        Tape identifier.
 * @param depen      Number of dependent variables (rows).
 * @param indep      Number of independent variables (columns).
 * @param repeat     If 0: compute sparsity and prepare seed (no numeric
 *                   recovery). If >0: perform numeric recovery using cached
 *                   seed.
 * @param basepoint  Array of independent values (required if tight
 *                   control-flow).
 * @param[in,out] nnz
 *                   Input/Output: when repeat==0 set by the routine to the
 *                   number of nonzeros discovered. When repeat!=0 must equal
 *                   the earlier computed number of nonzeros (consistency
 *                   check).
 * @param[out] rind  Pointer-to-pointer receiving row indices (coordinate
 *                   format).
 * @param[out] cind  Pointer-to-pointer receiving column indices (coordinate
 *                   format).
 * @param[out] values Pointer-to-pointer receiving numerical nonzero values.
 *
 * @return 0 or positive status on success, negative error code on failure.
 *
 * @note Memory ownership semantics for rind/cind/values mirror the underlying
 *       recovery routines: if user provides non-null pointers they will be used
 *       (usermem variants), otherwise unmanaged (allocator) variants are used
 *       and the caller is responsible for freeing the allocated memory.
 */
template <
    SparseMethod SM = SparseMethod::IndexDomains,
    CompressionMode CM = CompressionMode::Column,
    ControlFlowMode CFM = ControlFlowMode::Safe,
    BitPatternPropagationDirection BPPD = BitPatternPropagationDirection::Auto>
ADOLC_API int sparse_jac(short tag, int depen, int indep, int repeat,
                         const double *basepoint, int *nnz, unsigned int **rind,
                         unsigned int **cind, double **values) {
  using namespace detail;
  int ret_val = 0;
  if (repeat == 0)
    ret_val = buildJacPatternAndSeed<SM, CM, CFM, BPPD>(tag, depen, indep,
                                                        basepoint, nnz);
  if (ret_val < 0) {
    printf(" ADOL-C error in sparse_jac() \n");
    return ret_val;
  }
  return computeSparseJac<CM>(tag, depen, indep, basepoint, nnz, rind, cind,
                              values);
}

/**
 * @brief Computes the Jacobian sparsity pattern for an abs-normal form.
 *
 * This driver handles sparsity detection for functions represented in
 * *abs-normal form* — i.e., functions involving piecewise smooth structures
 * that depend on switching variables.
 *
 * Internally, it performs an index-domain propagation through the recorded
 * abs-normal tape.
 *
 * @param tag  ADOL-C tape identifier.
 * @param depen Number of dependent (output) variables.
 * @param indep Number of independent (input) variables.
 * @param numsw Number of switching variables in the abs-normal representation.
 * @param basepoint Pointer to basepoint values (independent variable array).
 * @param[out] compressedRowStorage
 *        Span over an array of row pointers, each representing one Jacobian
 *        row in Compressed Row Storage (CRS) format.
 *
 * @return Return code from `indopro_forward_absnormal()`:
 *         - `0` → success,
 *         - nonzero → warning or error code from ADOL-C.
 */
ADOLC_API int absnormal_jac_pat(short tag, int depen, int indep, int numsw,
                                const double *basepoint,
                                std::span<uint *> &compressedRowStorage);

/**
 * @brief Compute Hessian sparsity pattern (dispatch by control-flow mode).
 *
 * This function computes the sparsity pattern of the Hessian by dispatching to
 * the appropriate internal non-linear independent-index propagation driver
 * based on the compile-time ControlFlowMode template parameter.
 *
 * @tparam CFM ControlFlowMode controlling how control-flow is handled:
 *             - ControlFlowMode::OldTight : legacy tight control-flow handling
 *             - ControlFlowMode::OldSafe  : legacy safe control-flow handling
 *             - ControlFlowMode::Tight    : tight control-flow (evaluate
 * branches at basepoint)
 *             - ControlFlowMode::Safe     : safe / conservative control-flow
 *
 * @param tag                   ADOL-C tape identifier.
 * @param indep                 Number of independent variables (Hessian
 * dimension).
 * @param basepoint             Pointer to an array of length `indep` with the
 *                              evaluation point used when tight control-flow is
 * required.
 * @param[out] compressedRowStorage
 *                              Span of length `indep` where each element is a
 *                              pointer to a Compressed Row Storage (CRS) row
 * for the Hessian pattern. For row i:
 *                              - entry [i][0] is the count of indices stored,
 *                              - entry [i][1..] are the independents the output
 * i depend on.
 *
 * @return
 *    - >= 0 : Driver-specific non-negative code.
 *    - <  0 : Error code forwarded from the underlying driver.
 *
 * @note
 *  - The routine resets `compressedRowStorage` pointers before use.
 *  - Memory allocated for each row will be owned by the caller/tape.
 */
template <ControlFlowMode CFM>
ADOLC_API int hess_pat(short tag, int indep, const double *basepoint,
                       std::span<uint *> &compressedRowStorage) {
  detail::resetInput(compressedRowStorage);
  if constexpr (CFM == ControlFlowMode::OldTight)
    return nonl_ind_old_forward_tight(tag, 1, indep, basepoint,
                                      compressedRowStorage.data());
  else if (CFM == ControlFlowMode::OldSafe)
    return nonl_ind_old_forward_safe(tag, 1, indep, basepoint,
                                     compressedRowStorage.data());
  else if (CFM == ControlFlowMode::Tight)
    return nonl_ind_forward_tight(tag, 1, indep, basepoint,
                                  compressedRowStorage.data());
  else if (CFM == ControlFlowMode::Safe)
    return nonl_ind_forward_safe(tag, 1, indep, basepoint,
                                 compressedRowStorage.data());
}

/**
 * @brief Generate a seed matrix for compressed Hessian recovery.
 *
 * Produces a seed matrix for Hessian recovery using ColPack and
 * returns the unmanaged pointer to the seed via @p Seed and the compressed
 * dimension via @p p. The function uses ColPack's unmanaged API, so the
 * returned @p Seed memory is owned by ColPack and will be freed when the
 * ColPack graph object is destroyed (or as per ColPack's unmanaged semantics).
 *
 * @tparam RCM RecoveryMethod selecting the recovery strategy:
 *             - RecoveryMethod::Indirect : indirect recovery (acyclic strategy)
 *             - RecoveryMethod::Direct   : direct recovery (star strategy)
 *
 * @param n     Number of variables (dimension of Hessian).
 * @param HP    Span over CRS row pointers representing Hessian sparsity (HP).
 * @param[out] Seed  Pointer to the returned 2D seed matrix (ColPack-managed).
 * @param[out] p     Compressed dimension (number of seed columns/rows depending
 * on layout).
 *
 * @note The caller must respect ColPack unmanaged memory semantics for the
 * returned Seed.
 */
template <RecoveryMethod RCM>
ADOLC_API void generate_seed_hess(int n, std::span<uint *> HP, double ***Seed,
                                  int *p) {
  if constexpr (RCM == RecoveryMethod::Indirect)
    generateSeedHess(n, HP, Seed, p, "ACYCLIC_FOR_INDIRECT_RECOVERY");
  else if (RCM == RecoveryMethod::Direct)
    generateSeedHess(n, HP, Seed, p, "STAR");
}

namespace detail {

/**
 * @brief Build Hessian sparsity pattern and generate ColPack seed for recovery.
 *
 * This routine computes the sparsity pattern of the Hessian and constructs a
 * seed matrix for Hessian recovery using ColPack. It stores
 * pattern and seed/recovery related data inside the tape's `sHInfos()`
 * structure for reuse by subsequent numerical recovery calls.
 *
 * @tparam CFM ControlFlowMode used when computing the Hessian pattern:
 *             - ControlFlowMode::Safe  : conservative control-flow handling
 *             - ControlFlowMode::Tight : evaluate control-flow at @p basepoint
 * @tparam RCM RecoveryMethod selecting Hessian recovery strategy:
 *             - RecoveryMethod::Indirect : indirect (via two-phase recover)
 *             - RecoveryMethod::Direct   : direct recovery (star pattern)
 *
 * @param tag       ADOL-C tape identifier.
 * @param indep     Number of independent variables (dimension of Hessian).
 * @param basepoint Pointer to an array of length `indep` with the point at
 *                  which control-flow (if any) should be tested. May be
 *                  nullptr if not required by the chosen CFM (Safe vs Tight).
 * @param[out] nnz  Pointer to an integer where the number of nonzero Hessian
 *                  entries will be stored.
 *
 * @return
 *   - >= 0 : Success value forwarded from `hess_pat`
 *   - <  0 : Error code forwarded from `hess_pat`.
 *
 * @note
 *  - The computed sparsity pattern (HP) is stored in `tape.sHInfos().HP_`.
 *  - The generated seed matrix is stored (via ColPack) and its contents are
 *    copied into `tape.sHInfos().Xppp_`. The ColPack-owned seed memory will be
 *    freed when the ColPack graph object (`g_`) is destroyed.
 *  - Additional workspace arrays (Hcomp_, Xppp_, Yppp_, Zppp_, Upp_) are
 *    allocated on the tape for use by the numeric recovery routine.
 */
template <ControlFlowMode CFM, RecoveryMethod RCM>
int buildHessPatternAndSeed(short tag, int indep, const double *basepoint,
                            int *nnz) {
  ValueTape &tape = findTape(tag);
  int ret_val = -1;
  // Generate sparsity pattern, determine nnz, allocate memory
  tape.sHInfos().setHP(indep, std::vector<uint *>(indep));
  std::span<uint *> HPSpan = tape.sHInfos().HP_;
  // generate sparsity pattern
  ret_val = hess_pat<CFM>(tag, indep, basepoint, HPSpan);

  if (ret_val < 0) {
    printf(" ADOL-C error in sparse_hess() \n");
    return ret_val;
  }

  tape.sHInfos().indep_ = indep;
  tape.sHInfos().nnzIn_ = 0;

  for (uint i = 0; i < static_cast<uint>(indep); i++) {
    for (uint j = 1; j <= tape.sHInfos().HP_[i][0]; j++)
      if (tape.sHInfos().HP_[i][j] >= i)
        tape.sHInfos().nnzIn_++;
  }

  *nnz = tape.sHInfos().nnzIn_;

  // compute seed matrix => ColPack library

  // Seed is handled by ColPack
  double **Seed = nullptr;
  tape.sHInfos().initColoring(indep);
  if constexpr (RCM == RecoveryMethod::Indirect)
    tape.sHInfos().generateSeedHess(&Seed, "ACYCLIC_FOR_INDIRECT_RECOVERY");
  else if (RCM == RecoveryMethod::Direct)
    tape.sHInfos().generateSeedHess(&Seed, "STAR");

  // data might still be allocated, ensure that its not leaked
  myfree2(tape.sHInfos().Hcomp_);
  myfree3(tape.sHInfos().Xppp_);
  myfree3(tape.sHInfos().Yppp_);
  myfree3(tape.sHInfos().Zppp_);
  myfree2(tape.sHInfos().Upp_);

  tape.sHInfos().Hcomp_ = myalloc2(indep, tape.sHInfos().p_);
  tape.sHInfos().Xppp_ = myalloc3(indep, tape.sHInfos().p_, 1);

  for (int i = 0; i < indep; i++)
    for (int l = 0; l < tape.sHInfos().p_; l++)
      tape.sHInfos().Xppp_[i][l][0] = Seed[i][l];

  tape.sHInfos().Yppp_ = myalloc3(1, tape.sHInfos().p_, 1);
  tape.sHInfos().Zppp_ = myalloc3(tape.sHInfos().p_, indep, 2);
  tape.sHInfos().Upp_ = myalloc2(1, 2);
  tape.sHInfos().Upp_[0][0] = 1;
  tape.sHInfos().Upp_[0][1] = 0;
  return ret_val;
}

/**
 * @brief Compute sparse Hessian using precomputed seed/recovery info.
 *
 * Executes compressed forward/reverse evaluations followed by ColPack recovery
 * to produce the Hessian in coordinate (rind, cind, values) form. Supports
 * both indirect and direct recovery methods (template parameter RCM).
 *
 * @tparam RCM RecoveryMethod controlling the chosen ColPack recovery routine.
 *
 * @param tag       ADOL-C tape identifier.
 * @param indep     Number of independent variables (dimension of Hessian).
 * @param basepoint Pointer to an array of length `indep` with evaluation point.
 * @param nnz       Expected number of nonzeros (must equal the value
 *                  computed and stored by `buildHessPatternAndSeed`). If not,
 *                  the function returns error `-3`.
 * @param[out] rind Pointer-to-pointer that will receive the row indices of the
 *                 coordinate-format Hessian (allocated or user-provided).
 * @param[out] cind Pointer-to-pointer that will receive the column indices.
 * @param[out] values Pointer-to-pointer that will receive the numerical values.
 *
 * @return
 *   - >= 0 : Success (driver-specific non-negative codes).
 *   - <  0 : Error code from internal ADOL-C calls.
 *
 * @pre
 *  - `buildHessPatternAndSeed<...>` must have been called previously to
 *    populate `tape.sHInfos()` (seed, graph `g_`, hr_, workspace arrays).
 *
 * @note
 *  - The routine uses `hov_wk_forward` + `hos_ov_reverse` to compute compressed
 *    Hessian blocks and then invokes ColPack's recover routines. Both usermem
 *    (if rind/cind/values are non-null) and unmanaged variants are supported.
 *  - On success, caller is responsible for freeing memory returned by the
 *    unmanaged recovery functions (if used).
 */
template <RecoveryMethod RCM>
int computeSparseHess(short tag, int indep, const double *basepoint, int *nnz,
                      unsigned int **rind, unsigned int **cind,
                      double **values) {
  ValueTape &tape = findTape(tag);
  if (tape.sHInfos().Upp_ == nullptr) {
    printf(" ADOL-C error in sparse_hess():"
           " First call with repeat = 0 \n");
    return -3;
  }

  if (tape.sHInfos().nnzIn_ != *nnz) {
    printf(" ADOL-C error in sparse_hess():"
           " Number of nonzeros not consistent,"
           " new call with repeat = 0 \n");
    return -3;
  }

  //     this is the most efficient variant. However, there was somewhere a
  //     bug in hos_ov_reverse
  double y = 0.0;
  int ret_val =
      hov_wk_forward(tag, 1, indep, 1, 2, tape.sHInfos().p_, basepoint,
                     tape.sHInfos().Xppp_, &y, tape.sHInfos().Yppp_);
  MINDEC(ret_val, hos_ov_reverse(tag, 1, indep, 1, tape.sHInfos().p_,
                                 tape.sHInfos().Upp_, tape.sHInfos().Zppp_));

  for (int i = 0; i < tape.sHInfos().p_; ++i)
    for (int l = 0; l < indep; ++l)
      tape.sHInfos().Hcomp_[l][i] = tape.sHInfos().Zppp_[i][l][1];

  if (*values != nullptr && *rind != nullptr && *cind != nullptr) {
    // everything is preallocated, we assume correctly
    // call usermem versions
    if constexpr (RCM == RecoveryMethod::Indirect)
      tape.sHInfos().indirectRecoverUserMem(rind, cind, values);
    else if (RCM == RecoveryMethod::Direct)
      tape.sHInfos().directRecoverUserMem(rind, cind, values);
  } else {
    // at least one of rind cind values is not allocated, deallocate others
    // and call unmanaged versions
    if (*values != nullptr) {
      delete[] *values;
      *values = nullptr;
    }
    if (*rind != nullptr) {
      delete[] *rind;
      *rind = nullptr;
    }
    if (*cind != nullptr) {
      delete[] *cind;
      *cind = nullptr;
    }
    if constexpr (RCM == RecoveryMethod::Indirect)
      tape.sHInfos().indirectRecover(rind, cind, values);
    else if (RCM == RecoveryMethod::Direct)
      tape.sHInfos().directRecover(rind, cind, values);
  }
  return ret_val;
}

} // namespace detail

/**
 * @brief High-level API to compute a sparse Hessian.
 *
 * Coordinates Hessian sparsity pattern generation (when `repeat == 0`), seed
 * generation and caching, and the numeric recovery step. For `repeat == 0`
 * the function computes and caches pattern/seed info; for subsequent calls
 * it performs the numeric recovery using cached state.
 *
 * @tparam CFM ControlFlowMode used when computing the Hessian pattern.
 * @tparam RCM RecoveryMethod used by ColPack to perform recovery.
 *
 * @param tag        ADOL-C tape identifier.
 * @param indep      Number of independent variables (Hessian dimension).
 * @param repeat     If 0: compute pattern and seed (cache them). If >0: perform
 * @param basepoint  Basepoint for tight control-flow (may be nullptr if not
 *                  required).
 * @param[in,out] nnz
 *                   On entry: expected number of nonzeros when repeat != 0.
 *                   On exit: set to the number of nonzeros when repeat == 0.
 * @param[out] rind  Pointer-to-pointer to receive row indices (coordinate
 * format).
 * @param[out] cind  Pointer-to-pointer to receive column indices.
 * @param[out] values Pointer-to-pointer to receive numerical values.
 *
 * @return
 *   - >= 0 : Success (driver-specific non-negative codes).
 *   - <  0 : Error code (forwarded from pattern/compute routines).
 *
 * @note Memory ownership semantics for rind/cind/values match the underlying
 * recovery calls: if user provided non-null pointers, user memory is used
 * (usermem variants), otherwise the unmanaged variant allocates memory and the
 * caller must free it.
 */
template <ControlFlowMode CFM, RecoveryMethod RCM>
ADOLC_API int
sparse_hess(short tag, int indep, int repeat, const double *basepoint, int *nnz,
            unsigned int **rind, unsigned int **cind, double **values) {
  using namespace detail;
  int ret_val = 0;
  if (repeat == 0)
    ret_val = buildHessPatternAndSeed<CFM, RCM>(tag, indep, basepoint, nnz);
  if (ret_val < 0) {
    printf(" ADOL-C error in sparse_hess() \n");
    return ret_val;
  }
  return computeSparseHess<RCM>(tag, indep, basepoint, nnz, rind, cind, values);
}

/****************************************************************************/

} // namespace ADOLC::Sparse
#endif // ADOLC_SPARSE_DRIVERS_H
