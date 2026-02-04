#include <adolc/internal/common.h>
#include <adolc/sparse/sparse_options.h>
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/sparse/sparsedrivers_c.h>
#include <cstdio>
#include <span>

using ADOLC::Sparse::BitPatternPropagationDirection;
using ADOLC::Sparse::CompressionMode;
using ADOLC::Sparse::ControlFlowMode;
using ADOLC::Sparse::RecoveryMethod;
using ADOLC::Sparse::SparseMethod;

/// Helpers to map the sparse options
namespace detail {

inline SparseMethod parseSM(const int *options) {
  return (options[0] == 1) ? SparseMethod::BitPattern
                           : SparseMethod::IndexDomains;
}

inline ControlFlowMode parseJacCFM(const int *options) {
  return (options[1] == 1) ? ControlFlowMode::Tight : ControlFlowMode::Safe;
}

inline ControlFlowMode parseHessCFM(int option) {
  switch (option) {
  case 0:
    return ControlFlowMode::Safe;
  case 1:
    return ControlFlowMode::Tight;
  case 2:
    return ControlFlowMode::OldSafe;
  case 3:
    return ControlFlowMode::OldTight;
  }
  return ControlFlowMode::Safe;
}

inline BitPatternPropagationDirection parseBPPD(const int *options) {
  switch (options[2]) {
  case 1:
    return BitPatternPropagationDirection::Forward;
  case 2:
    return BitPatternPropagationDirection::Reverse;
  default:
    return BitPatternPropagationDirection::Auto;
  }
  return BitPatternPropagationDirection::Auto;
}

inline CompressionMode parseCM_(int option) {
  return (option == 1) ? CompressionMode::Row : CompressionMode::Column;
}

inline CompressionMode parseCM(const int *options) {
  return parseCM_(options[3]);
}

inline RecoveryMethod parseRM_(int option) {
  return (option == 1) ? RecoveryMethod::Direct : RecoveryMethod::Indirect;
}

inline RecoveryMethod parseRM(const int *options) {
  return parseRM_(options[1]);
}

template <SparseMethod SM, ControlFlowMode CFM,
          BitPatternPropagationDirection BPPD>
int call_jac_pat(short tag, int m, int n, const double *x, unsigned int **JP) {
  std::span<unsigned int *> jpSpan(JP, static_cast<size_t>(m));
  return ADOLC::Sparse::jac_pat<SM, CFM, BPPD>(tag, m, n, x, jpSpan);
}

template <ControlFlowMode CFM>
int call_hess_pat(short tag, int n, const double *x, unsigned int **HP) {
  std::span<unsigned int *> hpSpan(HP, static_cast<size_t>(n));
  return ADOLC::Sparse::hess_pat<CFM>(tag, n, x, hpSpan);
}

template <CompressionMode CM>
void call_generate_seed_jac(int m, int n, unsigned int **JP, double ***S,
                            int *p) {
  std::span<unsigned int *> jpSpan(JP, static_cast<size_t>(m));
  ADOLC::Sparse::generate_seed_jac<CM>(m, n, jpSpan, S, p);
}

template <RecoveryMethod RM>
void call_generate_seed_hess(int n, unsigned int **HP, double ***S, int *p) {
  std::span<unsigned int *> hpSpan(HP, static_cast<size_t>(n));
  return ADOLC::Sparse::generate_seed_hess<RM>(n, hpSpan, S, p);
}

template <SparseMethod SM, CompressionMode CM, ControlFlowMode CFM,
          BitPatternPropagationDirection BPPD>
int call_sparse_jac(short tag, int m, int n, int repeat, const double *x,
                    int *nnz, unsigned int **rind, unsigned int **cind,
                    double **values) {
  return ADOLC::Sparse::sparse_jac<SM, CM, CFM, BPPD>(tag, m, n, repeat, x, nnz,
                                                      rind, cind, values);
}

template <ControlFlowMode CFM, RecoveryMethod RM>
int call_sparse_hess(short tag, int n, int repeat, const double *x, int *nnz,
                     unsigned int **rind, unsigned int **cind,
                     double **values) {
  return ADOLC::Sparse::sparse_hess<CFM, RM>(tag, n, repeat, x, nnz, rind, cind,
                                             values);
}

} // namespace detail

extern "C" {

int jac_pat(short tag, int m, int n, const double *x, unsigned int **JP,
            int *options) {
  using namespace detail;
  const auto sm = parseSM(options);
  const auto cfm = parseJacCFM(options);
  const auto bpdd = parseBPPD(options);

  // Dispatch to templates. Keep behavior consistent with old options.
  if (sm == SparseMethod::IndexDomains) {
    if (cfm == ControlFlowMode::Tight) {
      return call_jac_pat<SparseMethod::IndexDomains, ControlFlowMode::Tight,
                          BitPatternPropagationDirection::Auto>(tag, m, n, x,
                                                                JP);
    } else {
      return call_jac_pat<SparseMethod::IndexDomains, ControlFlowMode::Safe,
                          BitPatternPropagationDirection::Auto>(tag, m, n, x,
                                                                JP);
    }
  } else {
    if (cfm == ControlFlowMode::Tight) {
      if (bpdd == BitPatternPropagationDirection::Forward)
        return call_jac_pat<SparseMethod::BitPattern, ControlFlowMode::Tight,
                            BitPatternPropagationDirection::Forward>(tag, m, n,
                                                                     x, JP);
      if (bpdd == BitPatternPropagationDirection::Reverse)
        return call_jac_pat<SparseMethod::BitPattern, ControlFlowMode::Tight,
                            BitPatternPropagationDirection::Reverse>(tag, m, n,
                                                                     x, JP);
      return call_jac_pat<SparseMethod::BitPattern, ControlFlowMode::Tight,
                          BitPatternPropagationDirection::Auto>(tag, m, n, x,
                                                                JP);
    } else {
      if (bpdd == BitPatternPropagationDirection::Forward)
        return call_jac_pat<SparseMethod::BitPattern, ControlFlowMode::Safe,
                            BitPatternPropagationDirection::Forward>(tag, m, n,
                                                                     x, JP);
      if (bpdd == BitPatternPropagationDirection::Reverse)
        return call_jac_pat<SparseMethod::BitPattern, ControlFlowMode::Safe,
                            BitPatternPropagationDirection::Reverse>(tag, m, n,
                                                                     x, JP);
      return call_jac_pat<SparseMethod::BitPattern, ControlFlowMode::Safe,
                          BitPatternPropagationDirection::Auto>(tag, m, n, x,
                                                                JP);
    }
  }
}

int hess_pat(short tag, int n, const double *x, unsigned int **HP,
             int *option) {
  using namespace detail;
  const auto cfm = parseHessCFM(option[0]);
  switch (cfm) {
  case ControlFlowMode::Safe:
    return call_hess_pat<ControlFlowMode::Safe>(tag, n, x, HP);
  case ControlFlowMode::Tight:
    return call_hess_pat<ControlFlowMode::Tight>(tag, n, x, HP);
  case ControlFlowMode::OldSafe:
    return call_hess_pat<ControlFlowMode::OldSafe>(tag, n, x, HP);
  case ControlFlowMode::OldTight:
    return call_hess_pat<ControlFlowMode::OldTight>(tag, n, x, HP);
  }
  return -1;
}

void generate_seed_jac(int m, int n, unsigned int **JP, double ***S, int *p,
                       int *option) {
  using namespace detail;
  const auto cm = parseCM_(option[0]);
  if (cm == CompressionMode::Column)
    return call_generate_seed_jac<CompressionMode::Column>(m, n, JP, S, p);
  else if (cm == CompressionMode::Row)
    return call_generate_seed_jac<CompressionMode::Row>(m, n, JP, S, p);
}

void generate_seed_hess(int n, unsigned int **HP, double ***S, int *p,
                        int *option) {
  using namespace detail;
  const auto rm = parseRM_(option[0]);
  if (rm == RecoveryMethod::Indirect)
    return call_generate_seed_hess<RecoveryMethod::Indirect>(n, HP, S, p);
  else if (rm == RecoveryMethod::Direct)
    return call_generate_seed_hess<RecoveryMethod::Direct>(n, HP, S, p);
}

int sparse_jac(short tag, int m, int n, int repeat, const double *x, int *nnz,
               unsigned int **rind, unsigned int **cind, double **values,
               int *options) {
  using namespace detail;
  const auto sm = parseSM(options);
  const auto cfm = parseJacCFM(options);
  const auto bpdd = parseBPPD(options);
  const auto cm = parseCM(options);

  // IMPORTANT: preserve old "allocate on repeat==0" expectation:
  // - If repeat==0, users expect this driver to allocate rind/cind/values.
  //   The C++ implementation already does "unmanaged" when pointers are null,
  //   so we enforce that by nulling them here.
  if (repeat == 0) {
    if (rind)
      *rind = nullptr;
    if (cind)
      *cind = nullptr;
    if (values)
      *values = nullptr;
  }

  // Dispatch all combinations that matter.
  // IndexDomains ignores bpdd.
  if (sm == SparseMethod::IndexDomains) {
    if (cm == CompressionMode::Row) {
      if (cfm == ControlFlowMode::Tight)
        return call_sparse_jac<SparseMethod::IndexDomains, CompressionMode::Row,
                               ControlFlowMode::Tight,
                               BitPatternPropagationDirection::Auto>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      return call_sparse_jac<SparseMethod::IndexDomains, CompressionMode::Row,
                             ControlFlowMode::Safe,
                             BitPatternPropagationDirection::Auto>(
          tag, m, n, repeat, x, nnz, rind, cind, values);
    } else {
      if (cfm == ControlFlowMode::Tight)
        return call_sparse_jac<SparseMethod::IndexDomains,
                               CompressionMode::Column, ControlFlowMode::Tight,
                               BitPatternPropagationDirection::Auto>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      return call_sparse_jac<SparseMethod::IndexDomains,
                             CompressionMode::Column, ControlFlowMode::Safe,
                             BitPatternPropagationDirection::Auto>(
          tag, m, n, repeat, x, nnz, rind, cind, values);
    }
  }

  if (cm == CompressionMode::Row) {
    if (cfm == ControlFlowMode::Tight) {
      if (bpdd == BitPatternPropagationDirection::Forward)
        return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Row,
                               ControlFlowMode::Tight,
                               BitPatternPropagationDirection::Forward>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      if (bpdd == BitPatternPropagationDirection::Reverse)
        return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Row,
                               ControlFlowMode::Tight,
                               BitPatternPropagationDirection::Reverse>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Row,
                             ControlFlowMode::Tight,
                             BitPatternPropagationDirection::Auto>(
          tag, m, n, repeat, x, nnz, rind, cind, values);
    } else {
      if (bpdd == BitPatternPropagationDirection::Forward)
        return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Row,
                               ControlFlowMode::Safe,
                               BitPatternPropagationDirection::Forward>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      if (bpdd == BitPatternPropagationDirection::Reverse)
        return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Row,
                               ControlFlowMode::Safe,
                               BitPatternPropagationDirection::Reverse>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Row,
                             ControlFlowMode::Safe,
                             BitPatternPropagationDirection::Auto>(
          tag, m, n, repeat, x, nnz, rind, cind, values);
    }
  } else { // Column
    if (cfm == ControlFlowMode::Tight) {
      if (bpdd == BitPatternPropagationDirection::Forward)
        return call_sparse_jac<SparseMethod::BitPattern,
                               CompressionMode::Column, ControlFlowMode::Tight,
                               BitPatternPropagationDirection::Forward>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      if (bpdd == BitPatternPropagationDirection::Reverse)
        return call_sparse_jac<SparseMethod::BitPattern,
                               CompressionMode::Column, ControlFlowMode::Tight,
                               BitPatternPropagationDirection::Reverse>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Column,
                             ControlFlowMode::Tight,
                             BitPatternPropagationDirection::Auto>(
          tag, m, n, repeat, x, nnz, rind, cind, values);
    } else {
      if (bpdd == BitPatternPropagationDirection::Forward)
        return call_sparse_jac<SparseMethod::BitPattern,
                               CompressionMode::Column, ControlFlowMode::Safe,
                               BitPatternPropagationDirection::Forward>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      if (bpdd == BitPatternPropagationDirection::Reverse)
        return call_sparse_jac<SparseMethod::BitPattern,
                               CompressionMode::Column, ControlFlowMode::Safe,
                               BitPatternPropagationDirection::Reverse>(
            tag, m, n, repeat, x, nnz, rind, cind, values);
      return call_sparse_jac<SparseMethod::BitPattern, CompressionMode::Column,
                             ControlFlowMode::Safe,
                             BitPatternPropagationDirection::Auto>(
          tag, m, n, repeat, x, nnz, rind, cind, values);
    }
  }
}

int sparse_hess(short tag, int n, int repeat, const double *x, int *nnz,
                unsigned int **rind, unsigned int **cind, double **values,
                int *options) {
  using namespace detail;
  const auto cfm = parseHessCFM(options[0]);
  const auto rm = parseRM(options);

  if (repeat == 0) {
    if (rind)
      *rind = nullptr;
    if (cind)
      *cind = nullptr;
    if (values)
      *values = nullptr;
  }

  switch (cfm) {
  case ControlFlowMode::Tight:
    if (rm == RecoveryMethod::Direct)
      return call_sparse_hess<ControlFlowMode::Tight, RecoveryMethod::Direct>(
          tag, n, repeat, x, nnz, rind, cind, values);
    return call_sparse_hess<ControlFlowMode::Tight, RecoveryMethod::Indirect>(
        tag, n, repeat, x, nnz, rind, cind, values);

  case ControlFlowMode::Safe:
    if (rm == RecoveryMethod::Direct)
      return call_sparse_hess<ControlFlowMode::Safe, RecoveryMethod::Direct>(
          tag, n, repeat, x, nnz, rind, cind, values);
    return call_sparse_hess<ControlFlowMode::Safe, RecoveryMethod::Indirect>(
        tag, n, repeat, x, nnz, rind, cind, values);

  case ControlFlowMode::OldTight:
    if (rm == RecoveryMethod::Direct)
      return call_sparse_hess<ControlFlowMode::OldTight,
                              RecoveryMethod::Direct>(tag, n, repeat, x, nnz,
                                                      rind, cind, values);
    return call_sparse_hess<ControlFlowMode::OldTight,
                            RecoveryMethod::Indirect>(tag, n, repeat, x, nnz,
                                                      rind, cind, values);
  case ControlFlowMode::OldSafe:
    if (rm == RecoveryMethod::Direct)
      return call_sparse_hess<ControlFlowMode::OldSafe, RecoveryMethod::Direct>(
          tag, n, repeat, x, nnz, rind, cind, values);
    return call_sparse_hess<ControlFlowMode::OldSafe, RecoveryMethod::Indirect>(
        tag, n, repeat, x, nnz, rind, cind, values);
  }
  return -1;
}
} // extern "C"
