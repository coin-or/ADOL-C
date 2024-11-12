#ifndef ARRAY_HANDLER_H
#define ARRAY_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif
const double getindex_vec(const double *A, const int row);
void setindex_vec(double *A, const double val, const int row);
const double getindex_mat(const double **A, const int row, const int col);
void setindex_mat(double **A, const double val, const int row, const int col);
const double **getindex_ten(const double ***A, const int dim);
const double getindex_tens(const double ***A, const int dim, const int row,
                           const int col);
void setindex_tens(double ***A, const double val, const int dim, const int row,
                   const int col);
short **alloc_short_mat(const int row, const int col);
void free_short_mat(short **s);
#ifdef __cplusplus
}
#endif

#endif // ARRAY_HANDLER_H