#include "array_handler.h"

/*
Setter and getter for vectors, matrizes and tensors.
*/
extern "C" {
const double getindex_vec(const double *A, const int row) { return A[row]; }
void setindex_vec(double *A, const double val, const int row) { A[row] = val; }

const double getindex_mat(const double **A, const int row, const int col) {
  return A[row][col];
}
void setindex_mat(double **A, const double val, const int row, const int col) {
  A[row][col] = val;
}
const double **getindex_ten(const double ***A, const int dim) { return A[dim]; }
const double getindex_tens(const double ***A, const int dim, const int row,
                           const int col) {
  return A[dim][row][col];
}
void setindex_tens(double ***A, const double val, const int dim, const int row,
                   const int col) {
  A[dim][row][col] = val;
}

short **alloc_short_mat(const int rows, const int cols) {
  short **s = new short *[rows];
  for (int i = 0; i < rows; i++) {
    s[i] = new short[cols];
  }
  return s;
}
void free_short_mat(short **s) { delete s; }
}