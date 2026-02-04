#include <adolc/adolcerror.h>
#include <adolc/sparse/sparse_fo_rev.h>
#include <adolc/sparse/sparse_options.h>
#include <vector>

int ADOLC::Sparse::forward(short tag, int m, int n, int p, double *x,
                           bitword_t **X, double *y, bitword_t **Y, char mode) {
  std::vector<bitword_t *> Xv(X, X + n);
  std::vector<bitword_t *> Yv(Y, Y + m);

  if (mode == 0)
    return ADOLC::Sparse::forward<ADOLC::Sparse::ControlFlowMode::Safe>(
        tag, m, n, p, x, Xv, y, Yv);
  else if (mode == 1)
    return ADOLC::Sparse::forward<ADOLC::Sparse::ControlFlowMode::Tight>(
        tag, m, n, p, x, Xv, y, Yv);
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);
  return -1;
}

int ADOLC::Sparse::forward(short tag, int m, int n, int p, bitword_t **X,
                           bitword_t **Y, char mode) {
  if (mode == 0)
    return ADOLC::Sparse::forward(tag, m, n, p, nullptr, X, nullptr, Y, mode);
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);
  return -1;
}

int ADOLC::Sparse::reverse(short tag, int m, int n, int q, bitword_t **U,
                           bitword_t **Z, char mode) {
  std::vector<bitword_t *> Uv(U, U + q);
  std::vector<bitword_t *> Zv(Z, Z + q);

  if (mode == 0)
    ADOLC::Sparse::reverse<ADOLC::Sparse::ControlFlowMode::Safe>(tag, m, n, q,
                                                                 &Uv, &Zv);
  else if (mode == 1)
    ADOLC::Sparse::reverse<ADOLC::Sparse::ControlFlowMode::Tight>(tag, m, n, q,
                                                                  &Uv, &Zv);
  else
    ADOLCError::fail(ADOLCError::ErrorType::SPARSE_BAD_MODE, CURRENT_LOCATION);
  return -1;
}
