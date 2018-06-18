

#include <adolc/adolc_openmp.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include "taping_p.h"

/* Write thread local part of Jacobian to global Jacobian. */
void writeLocJac2globJac(double** loc_jacobian, double** jacobian, int depen,
                         int part, int loc_start)
{
  int j, k;
  for (int i = 0; i < depen; ++i) {
    for (k = 0, j = loc_start; j < loc_start+part; ++k, ++j)
      jacobian[i][j] = loc_jacobian[i][k];
  }
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
void calcPartitioning(int* part, int* loc_start, int num_threads, int N)
{
  int tmppart = N / num_threads;
  for (int i = 0; i < num_threads; ++i)
    part[i] = tmppart;
  for (int i = 0; i < N%num_threads; ++i)
    ++part[i];

  loc_start[0] = 0;
  for (int i = 1; i < num_threads; ++i)
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
  int num_threads = 1;
#ifdef _OPENMP
  if (omp_get_max_threads() > depen) {
    num_threads = depen;
    omp_set_num_threads(num_threads);
  }
  else {
    num_threads = omp_get_max_threads();
  }
  fprintf(DIAG_OUT,"ADOL-C info: Parallel Jacobian method uses %i threads.\n",
          num_threads);
#endif

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
    // Error handling
    if (rc[0] < 0)
      return rc[0];
    else {
      for (int i = 1; i < num_threads; ++i)
        rc[i] = rc[0];
    }
    calcPartitioning(loc_part, loc_start, num_threads, depen);
    // Debug
    // for (i = 0; i < num_threads; ++i)
    //   printf("num_threads= %d, loc_part[%d] = %d , loc_start[%d] = %d \n",
    //          num_threads, i, loc_part[i], i, loc_start[i]);
  }

  // Dummy parallel region prevents undefined behaviour w.r.t. threadprivate + copyin.
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
    writeLocJac2globJac(loc_jac, jacobian, depen, loc_part[myid], loc_start[myid]);
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

  for (int i = 0; i < num_threads; ++i)
    MINDEC(rcg, rc[i]);

  delete rc;
  return rcg;
}
