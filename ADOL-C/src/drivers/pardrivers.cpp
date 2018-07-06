

#include <adolc/adolc_openmp.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include "taping_p.h"

#include <iostream>
#include <assert.h>

static int get_thread_num()
{
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

static uint get_num_threads(int num)
{
  uint num_threads = 1;
#ifdef _OPENMP
  if (omp_get_max_threads() > num) {
    num_threads = num;
    omp_set_num_threads(num);
  }
  else {
    num_threads = omp_get_max_threads();
  }
#endif
  return num_threads;
}

/**
 * Returns the submatrix S containing nrows consecutive rows and ncols consecutive
 * columns from given matrix U starting from row_st row and col_st column,
 * respectively.
 *
 * U: (m, n)
 * S: (nrows, ncols)
 */
static double** get_submat(const uint m, const uint n,
                           const uint st_row, const uint nrows,
                           const uint st_col, const uint ncols,
                           double** U)
{
  // Do not touch rows and columns behind memory
  assert((st_row+nrows) <= m);
  assert((st_col+ncols) <= n);
  double** loc_mat = myalloc2(nrows, ncols);
  for (uint i = st_row, ii = 0; i < (st_row+nrows); ++i, ++ii)
    for (uint j = st_col, jj = 0; j < (st_col+ncols); ++j, ++jj)
      loc_mat[ii][jj] = U[i][j];

  return loc_mat;
}

/* Verallgemeinerung: write block back, not just rows or columns */
/*
 * U : m, n
 */
static void writeLocMat2globMat(const int m, const int n,
                                const uint st_row, const uint nrows,
                                const uint st_col, const uint ncols,
                                double** loc_jac, double** jacobian)
{
  for (uint i = 0, ii = st_row; ii < (st_row+nrows); ++i, ++ii)
    for (uint j = 0, jj = st_col; jj < (st_col+ncols); ++j, ++jj)
      jacobian[ii][jj] = loc_jac[i][j];
}

/* Jacobian Partitioning
 *
 * Calculates partitioning of the Jacobian matrix (or a general matrix) according
 * to the given number of threads and parameter N. The matrix is partitioned into
 * contiguous strips of
 *    - columns, if N represents the matrix's number of columns
 *    - rows, if N represents the matrix's number of columns.
 * If N%num_threads = r > 0, i.e., the first r threads are assigned an additional
 * row or column, respectively.
 */
void calcPartitioning(int* part, int* loc_start, uint num_threads, int N)
{
  int tmppart = N / num_threads;
  for (uint i = 0; i < num_threads; ++i)
    part[i] = tmppart;
  for (uint i = 0; i < N%num_threads; ++i)
    ++part[i];

  loc_start[0] = 0;
  for (uint i = 1; i < num_threads; ++i)
    loc_start[i] = loc_start[i-1] + part[i-1];
}

/*--------------------------------------------------------------------------*/
/*                                                        parallel jacobian */
/* par_jacobian(tag, m, n, x[n], J[m][n])                                   */
/*--------------------------------------------------------------------------*/
int par_jacobian(short tag,
                 int depen,
                 int indep,
                 const double* argument,
                 double** jacobian)
{
  uint num_threads = get_num_threads(depen);
  fprintf(DIAG_OUT,"ADOL-C info: par_jacobian uses %i OpenMP-threads.\n",
          num_threads);

  int rcg = 0;
  int* rc = new int[num_threads];
  // Array for return code.
  if (NULL == rc) {
    fail(ADOLC_MALLOC_FAILED);
    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }
  // loc_part[i] := # columns assigned to thread i
  int* loc_part = (int*)calloc(num_threads, sizeof(int));
  if (NULL == loc_part) {
      fail(ADOLC_MALLOC_FAILED);
      adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
  // loc_start[i] := column to start loc_part for thread i
  int* loc_start = (int*)calloc(num_threads, sizeof(int));
  if (NULL == loc_start) {
    fail(ADOLC_MALLOC_FAILED);
    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }

  // Forward sweep for reverse mode.
  if (indep/2 < depen)
    calcPartitioning(loc_part, loc_start, num_threads, indep);
  else {
    double* result = myalloc1(depen);
    rc[0] = zos_forward(tag, depen, indep, 1, argument, result);
    myfree1(result);
    // Error handling.
    if (rc[0] < 0) {
      free(loc_start);
      free(loc_part);
      return rc[0];
    }
    else {
      for (uint i = 1; i < num_threads; ++i)
        rc[i] = rc[0];
    }
    calcPartitioning(loc_part, loc_start, num_threads, depen);
    // Debug
    // for (i = 0; i < num_threads; ++i)
    //   printf("num_threads= %d, loc_part[%d] = %d , loc_start[%d] = %d \n",
    //          num_threads, i, loc_part[i], i, loc_start[i]);
  }

  // Dummy par. region prevents undef. behaviour w.r.t. threadprivate+copyin.
  // Leave it, even the compiler will encourage you to remove it!
  // Rationale: Reference threadprivate variable first, than copy value of
  //            master thread via copyin clause.
#pragma omp parallel
  {
    ADOLC_OpenMP;
  }

#ifdef _OPENMP
#pragma omp parallel default(none) shared(tag, rc, argument, jacobian, num_threads, loc_part, loc_start) \
                                   copyin(ADOLC_OpenMP) \
                                   firstprivate(depen, indep)
#endif
{
#ifdef _OPENMP
  int myid = omp_get_thread_num();
#else
  int myid = 0;
#endif

  double** loc_I = NULL;
  double** loc_jac;

  /* Over decomposition using for loop and chunks, i.e., more "blocks" than number of threads
   * would be equivalent to scalar modus. */
  /* Current implementation: number of threads = number of blocks */
  if (indep/2 < depen) {
    double* result = myalloc1(depen);
    /* n*m matrix, with ones from pth rows on the diagonal. Kind of a unit matrix. */
    loc_I = myallocI2nmp(indep, loc_part[myid], loc_start[myid]);
    /* Allocate thread local Jacobian of size depen*#local_cols (num of cols current thread comp.) */
    /* sum of #local_cols = indep */
    loc_jac = myalloc2(depen, loc_part[myid]);
    rc[myid] = fov_forward(tag,depen,indep, loc_part[myid], argument, loc_I, result, loc_jac);
    /* Each thread writes loc_jac into global Jacobian object. */
    writeLocMat2globMat(depen, indep, 0, depen, loc_start[myid], loc_part[myid], loc_jac, jacobian);
    /* free memory */
    myfreeI2nmp(indep, loc_part[myid], loc_start[myid], loc_I);
    myfree1(result);
  } else {
    /* n*m matrix, ab p-ten Zeile Einsen */
    loc_I = myallocI2nmp(loc_part[myid], depen, loc_start[myid]);
    loc_jac = (double**)calloc(loc_part[myid], sizeof(double*));
    for (int i = 0; i < loc_part[myid]; ++i)
      loc_jac[i] = jacobian[loc_start[myid]+i];

    MINDEC(rc[myid], fov_reverse(tag, depen, indep, loc_part[myid], loc_I, loc_jac));
    myfreeI2nmp(depen, loc_part[myid], loc_start[myid], loc_I);
  }
  myfree2(loc_jac);
} /* parallel */

#ifndef NDEBUG
  std::cout << "ADOL-C info: Return codes of fov_reverse() within par_jac()"
               " driver (Values < 0 indicate an error).\n";
  for (uint i = 0; i < num_threads; ++i)
    std::cout << "  Thread " << i << " : " << rc[i] << " \n";
#endif

  for (uint i = 0; i < num_threads; ++i)
    MINDEC(rcg, rc[i]);

  free(loc_start);
  free(loc_part);
  delete[] rc;

  return rcg;
}

/* [in] m, n, p, U
 * [out] Z
 *
 * Compute: Z = UJ(x_o)
 * whereas J \in R^{m,n}, U in R^{p,m} and Z \in R^{p,n}
 */
int par_mat_jac(short tag, int m, int n, int p, const double* argument,
                double** U, double** Z)
{
  uint num_threads = get_num_threads(p);
  fprintf(DIAG_OUT,"ADOL-C info: par_jac_mat uses %i OpenMP-threads.\n",
          num_threads);

  int ret;
  int* rc = new int[num_threads];
  double* result = myalloc1(m);
  ret = zos_forward(tag, m, n, 1, argument, result);
  myfree1(result);
  // Error? Clean up and return.
  if (0 > ret) {
    delete[] rc;
    return ret;
  }

  // loc_part[i] := # rows assigned to thread i
  int* loc_part = (int*)calloc(num_threads, sizeof(int));
  if (NULL == loc_part) {
    fail(ADOLC_MALLOC_FAILED);
    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }
  // loc_start[i] := row to start loc_part for thread i
  int* loc_start = (int*)calloc(num_threads, sizeof(int));
  if (NULL == loc_start) {
    fail(ADOLC_MALLOC_FAILED);
    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }
  // Partitioning of rows of U
  calcPartitioning(loc_part, loc_start, num_threads, p);

#ifdef _OPENMP
#pragma omp parallel default(none) shared(tag, rc, argument, U, Z, num_threads, loc_part, loc_start, p) \
                                   copyin(ADOLC_OpenMP) \
                                   firstprivate(m, n)
#endif
  {
    int myid = get_thread_num();
    double** loc_U;
    // If we have only 1 thread, we don't want allocate additional memory.
    if (1 < num_threads)
      //loc_U = get_mat_part_rows(p, m, loc_start[myid], loc_part[myid], U);
      loc_U = get_submat(p, m, loc_start[myid], loc_part[myid], 0, m, U);
    else
      loc_U = U;

    // Since the result is a block of rows, we can directly let the loc_Z[i] point to the
    // appropriate row in global Z
    double** loc_Z = (double**)calloc(loc_part[myid], sizeof(double*));
    for (int i = 0; i < loc_part[myid]; ++i)
      loc_Z[i] = Z[loc_start[myid]+i];

    rc[myid] = fov_reverse(tag, m, n, loc_part[myid], loc_U, loc_Z);

    // Clean up.
    myfree2(loc_Z);
    if (1 < num_threads)
      myfree2(loc_U);
  } // end parallel

  free(loc_start);
  free(loc_part);

#ifndef NDEBUG
  std::cout << "ADOL-C info: Return codes of fov_forward() within par_jac_mat()"
               " driver (Values < 0 indicate an error).\n";
  for (size_t i = 0; i < num_threads; ++i)
    std::cout << "  Thread " << i << " : " << rc[i] << " \n";
#endif

  int rcg = 0;
  for (uint i = 0; i < num_threads; ++i)
    MINDEC(rcg, rc[i]);

  delete[] rc;

  return rcg;
}

/* [in] tag, m, n, p, x, U
 * [out] Z
 *
 * Compute: Z = J(x_o)U
 * whereas J \in R^{m,n}, U in R^{n,p} and Z \in R^{m,p}
 */
int par_jac_mat(short tag, int m, int n, int p, const double* argument,
                double** U, double** Z)
{
  uint num_threads = get_num_threads(p);
  fprintf(DIAG_OUT,"ADOL-C info: par_jac_mat uses %i OpenMP-threads.\n",
          num_threads);

  int* rc = new int[num_threads];

  // loc_part[i] := # columns assigned to thread i
  int* loc_part = (int*)calloc(num_threads, sizeof(int));
  if (NULL == loc_part) {
    fail(ADOLC_MALLOC_FAILED);
    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }
  // loc_start[i] := column to start loc_part for thread i
  int* loc_start = (int*)calloc(num_threads, sizeof(int));
  if (NULL == loc_start) {
    fail(ADOLC_MALLOC_FAILED);
    adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }
  // Partitioning of columns of U
  calcPartitioning(loc_part, loc_start, num_threads, p);

  // Dummy par. region prevents undef. behaviour w.r.t. threadprivate+copyin.
  // Leave it, even the compiler will encourage you to remove it!
  // Rationale: Reference threadprivate variable first, than copy value of
  //            master thread via copyin clause.
#pragma omp parallel
  {
    ADOLC_OpenMP;
  }

#ifdef _OPENMP
#pragma omp parallel default(none) shared(tag, rc, argument, U, Z, num_threads, loc_part, loc_start, p) \
                                   copyin(ADOLC_OpenMP) \
                                   firstprivate(m, n)
#endif
  {
    int myid = get_thread_num();
    double* result = myalloc1(m);
    double** loc_U;
    // If we have only 1 thread, we don't want we allocate additional memory.
    if (1 < num_threads)
      loc_U = get_submat(n, p, 0, n, loc_start[myid], loc_part[myid], U);
    else
      loc_U = U;

    double** loc_Z = myalloc2(m, loc_part[myid]);
    rc[myid] = fov_forward(tag, m, n, loc_part[myid], argument, loc_U, result, loc_Z);
    /* Each thread writes loc_Z into global Z object. */
    writeLocMat2globMat(m, p, 0, m, loc_start[myid], loc_part[myid], loc_Z, Z);

    // Clean up.
    myfree1(result);
    myfree2(loc_Z);
    if (1 < num_threads)
      myfree2(loc_U);
  } // end parallel

  free(loc_start);
  free(loc_part);

#ifndef NDEBUG
  std::cout << "ADOL-C info: Return codes of fov_forward() within par_jac_mat()"
               " driver (Values < 0 indicate an error).\n";
  for (size_t i = 0; i < num_threads; ++i)
    std::cout << "  Thread " << i << " : " << rc[i] << " \n";
#endif

  int rcg = 0;
  for (uint i = 0; i < num_threads; ++i)
    MINDEC(rcg, rc[i]);

  delete[] rc;

  return rcg;
}

