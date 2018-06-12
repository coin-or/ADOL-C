/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/drivers.c
 Revision: $Id$
 Contents: Easy to use drivers for optimization and nonlinear equations
           (Implementation of the C/C++ callable interfaces).
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include "taping_p.h"
#include <math.h>

BEGIN_C_DECLS

/****************************************************************************/
/*                         DRIVERS FOR OPTIMIZATION AND NONLINEAR EQUATIONS */

/*--------------------------------------------------------------------------*/
/*                                                                 function */
/* function(tag, m, n, x[n], y[m])                                          */
int function(short tag,
             int m,
             int n,
             double* argument,
             double* result) {
    int rc= -1;

    rc= zos_forward(tag,m,n,0,argument,result);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 gradient */
/* gradient(tag, n, x[n], g[n])                                             */
int gradient(short tag,
             int n,
             const double* argument,
             double* result) {
    int rc= -1;
    double one = 1.0;

    rc = zos_forward(tag,1,n,1,argument,result);
    if(rc < 0)
        return rc;
    MINDEC(rc, fos_reverse(tag,1,n,&one,result));
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                          */
/* vec_jac(tag, m, n, repeat, x[n], u[m], v[n])                             */
int vec_jac(short tag,
            int m,
            int n,
            int repeat,
            double* argument,
            double* lagrange,
            double* row) {
    int rc= -1;
    double *y = NULL;

    if(!repeat) {
        y = myalloc1(m);
        rc = zos_forward(tag,m,n,1, argument, y);
        if(rc < 0) return rc;
    }
    MINDEC(rc, fos_reverse(tag,m,n,lagrange,row));
    if (!repeat) myfree1(y);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* jacobian(tag, m, n, x[n], J[m][n])                                       */

void printvec(const char* name, int n, double* v)
{
	int i;
	printf("%s \n", name);
	for (i =0; i<n;++i){
		printf(" %10.4f", v[i]);
	}
	printf("\n");
}

int jacobian(short tag,
             int depen,
             int indep,
             const double *argument,
             double **jacobian) {
    int rc;
    double *result, **I;

    result = myalloc1(depen);

    if (indep/2 < depen) {
        I = myallocI2(indep);
        rc = fov_forward(tag,depen,indep,indep,argument,I,result,jacobian);
        myfreeI2(indep, I);
    } else {
        I = myallocI2(depen);
        rc = zos_forward(tag,depen,indep,1,argument,result);
        if (rc < 0) return rc;

        // x1 Temporary output. Remove later.
#pragma omp master
        {
        	TapeInfos* tpi = getTapeInfos(tag);
        	printvec("taybuffer", tpi->numTays_Tape, tpi->tayBuffer);
        }

        MINDEC(rc,fov_reverse(tag,depen,indep,depen,I,jacobian));
        myfreeI2(depen, I);
    }

    myfree1(result);

    return rc;
}

/* Write thread local part of Jacobian to global Jacobian. */
void writeLocJac2globJac(double** loc_jacobian, double** jacobian, int depen,
                         int part, int loc_start)
{
  int i, j, k;
  for (i = 0; i < depen; ++i) {
    for (k = 0, j = loc_start; j < loc_start+part; ++k, ++j)
      jacobian[i][j] = loc_jacobian[i][k];
  }
}

/* Jacobian Partitioning
 *
 * Partitions the Jacobian into contiguous strips of columns according to the
 * number of threads available for parallel Jacobian computation.
 * If depen%num_threads = r > 0, i.e., the first r threads are assigned an
 * additional column.
 */
void calcPartitioning(int* part, int* loc_start, int num_threads, int depen)
{
  int i;
  int tmppart = depen / num_threads;
  for (i = 0; i < num_threads; ++i)
    part[i] = tmppart;
  for (i=0; i < depen%num_threads; ++i)
    ++part[i];

  loc_start[0] = 0;
  for (i=1; i < num_threads; ++i)
    loc_start[i] = loc_start[i-1] + part[i-1];
}

#include <adolc/adolc_openmp.h>
/*--------------------------------------------------------------------------*/
/*                                                        parallel jacobian */
/* par_jacobian(tag, m, n, x[n], J[m][n])                                   */
int par_jacobian(short tag,
                 int depen,
                 int indep,
                 const double *argument,
                 double **jacobian)
{
#ifdef _OPENMP
  int num_threads = omp_get_max_threads();
#else
  int num_threads = 1;
#endif

  int i;
  int rcg = 0;
  int* rc = malloc(sizeof *rc * num_threads);
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
    // Error
    if (rc[0] < 0)
      return rc[0];
    else {
      for (i = 1; i < num_threads; ++i)
        rc[i] = rc[0];
    }
    calcPartitioning(loc_part, loc_start, num_threads, depen);
    for (i = 0; i < num_threads; ++i)
      printf("num_threads= %d, loc_part[%d] = %d , loc_start[%d] = %d \n", num_threads, i, loc_part[i], loc_start[i]);
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
  printf("myid: %d indep: %d, depen: %d \n", myid, indep, depen);

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
    /* n * m matrix, ab p-ten Zeile Einsen */
    loc_I = myallocI2nmp(loc_part[myid], depen, loc_start[myid]);
    loc_jac = calloc(loc_part[myid], sizeof(double*));
    int i;
    for (i = 0; i < loc_part[myid]; ++i)
      loc_jac[i] = jacobian[loc_start[myid]+i];
    MINDEC(rc[myid], fov_reverse(tag, depen, indep, loc_part[myid], loc_I, loc_jac));
    myfreeI2nmp(depen, loc_part[myid], loc_start[myid], loc_I);
  }
  myfree2(loc_jac);
} /* parallel */

  for (i = 0; i < num_threads; ++i)
    MINDEC(rcg, rc[i]);
  return rcg;
}

/*--------------------------------------------------------------------------*/
/*                                                           large_jacobian */
/* large_jacobian(tag, m, n, k, x[n], y[m], J[m][n])                        */

int large_jacobian(short tag,
		   int depen,
		   int indep,
		   int runns,
		   double *argument,
		   double *result,
		   double **jacobian)
{
    int rc, dirs, i;
    double **I;

	I = myallocI2(indep);
    if (runns > indep) runns = indep;
    if (runns < 1)     runns = 1;
    dirs = indep / runns;
    if (indep % runns) ++dirs;
    for (i=0; i<runns-1; ++i) {
        rc = fov_offset_forward(tag, depen, indep, dirs, i * dirs, argument,
                I, result, jacobian);
    }
    dirs = indep - (runns-1) * dirs;
    rc = fov_offset_forward(tag, depen, indep, dirs, indep-dirs, argument,
		     I, result, jacobian);
    myfreeI2(indep, I);
    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  jac_vec */
/* jac_vec(tag, m, n, x[n], v[n], u[m]);                                    */
int jac_vec(short tag,
            int m,
            int n,
            double* argument,
            double* tangent,
            double* column) {
    int rc= -1;
    double *y;

    y = myalloc1(m);

    rc = fos_forward(tag, m, n, 0, argument, tangent, y, column);
    myfree1(y);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                 hess_vec */
/* hess_vec(tag, n, x[n], v[n], w[n])                                       */
int hess_vec(short tag,
             int n,
             double *argument,
             double *tangent,
             double *result) {
    double one = 1.0;
    return lagra_hess_vec(tag,1,n,argument,tangent,&one,result);
}

/*--------------------------------------------------------------------------*/
/*                                                                 hess_mat */
/* hess_mat(tag, n, q, x[n], V[n][q], W[n][q])                              */
int hess_mat(short tag,
             int n,
             int q,
             double *argument,
             double **tangent,
             double **result) {
    int rc;
    int i,j;
    double y;
    double*** Xppp;
    double*** Yppp;
    double*** Zppp;
    double**  Upp;

    Xppp = myalloc3(n,q,1);   /* matrix on right-hand side  */
    Yppp = myalloc3(1,q,1);   /* results of hos_wk_forward  */
    Zppp = myalloc3(q,n,2);   /* result of Up x H x XPPP */
    Upp  = myalloc2(1,2);     /* vector on left-hand side */

    for (i = 0; i < n; ++i)
        for (j = 0; j < q; ++j)
            Xppp[i][j][0] = tangent[i][j];

    Upp[0][0] = 1;
    Upp[0][1] = 0;

    rc = hov_wk_forward(tag, 1, n, 1, 2, q, argument, Xppp, &y, Yppp);
    MINDEC(rc, hos_ov_reverse(tag, 1, n, 1, q, Upp, Zppp));

    for (i = 0; i < q; ++i)
        for (j = 0; j < n; ++j)
            result[j][i] = Zppp[i][j][1];

    myfree2(Upp);
    myfree3(Zppp);
    myfree3(Yppp);
    myfree3(Xppp);

    return rc;
}

/*--------------------------------------------------------------------------*/
/*                                                                  hessian */
/* hessian(tag, n, x[n], lower triangle of H[n][n])                         */
/* uses Hessian-vector product                                              */
int hessian(short tag,
            int n,
            double* argument,
            double** hess) {
    int rc= 3;
    int i,j;
    double *v = myalloc1(n);
    double *w = myalloc1(n);
    for(i=0;i<n;i++) v[i] = 0;
    for(i=0;i<n;i++) {
        v[i] = 1;
        MINDEC(rc, hess_vec(tag, n, argument, v, w));
        if( rc < 0) {
            free((char *)v);
            free((char *) w);
            return rc;
        }
        for(j=0;j<=i;j++)
            hess[i][j] = w[j];
        v[i] = 0;
    }

    free((char *)v);
    free((char *) w);
    return rc;
    /* Note that only the lower triangle of hess is filled */
}

/*--------------------------------------------------------------------------*/
/*                                                                 hessian2 */
/* hessian2(tag, n, x[n], lower triangle of H[n][n])                        */
/* uses Hessian-matrix product                                              */
int hessian2(short tag,
             int n,
             double* argument,
             double** hess) {
    int rc;
    int i,j;

    double*** Xppp = myalloc3(n,n,1);   /* matrix on right-hand side  */
    double*   y    = myalloc1(1);       /* results of function evaluation */
    double*** Yppp = myalloc3(1,n,1);   /* results of hos_wk_forward  */
    double*** Zppp = myalloc3(n,n,2);   /* result of Up x H x XPPP */
    double**  Upp  = myalloc2(1,2);     /* vector on left-hand side */

    for (i=0; i<n; i++) {
        for (j=0;j<n;j++)
            Xppp[i][j][0] = 0;
        Xppp[i][i][0] = 1;
    }

    Upp[0][0] = 1;
    Upp[0][1] = 0;

    rc = hov_wk_forward(tag,1,n,1,2,n,argument,Xppp,y,Yppp);
    MINDEC(rc,hos_ov_reverse(tag,1,n,1,n,Upp,Zppp));

    for (i=0; i<n; i++)
        for (j=0;j<=i;j++)
            hess[i][j] = Zppp[i][j][1];

    myfree2(Upp);
    myfree3(Zppp);
    myfree3(Yppp);
    myfree1(y);
    myfree3(Xppp);
    return rc;
    /* Note that only the lower triangle of hess is filled */
}

/*--------------------------------------------------------------------------*/
/*                                                           lagra_hess_vec */
/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
int lagra_hess_vec(short tag,
                   int m,
                   int n,
                   double *argument,
                   double *tangent,
                   double *lagrange,
                   double *result) {
    int rc=-1;
    int i;
    int degree = 1;
    int keep = degree+1;
    double **X, *y, *y_tangent;

    X = myalloc2(n,2);
    y = myalloc1(m);
    y_tangent = myalloc1(m);

    rc = fos_forward(tag, m, n, keep, argument, tangent, y, y_tangent);

    if(rc < 0) return rc;

    MINDEC(rc, hos_reverse(tag, m, n, degree, lagrange, X));

    for(i = 0; i < n; ++i)
        result[i] = X[i][1];

    myfree1(y_tangent);
    myfree1(y);
    myfree2(X);

    return rc;
}

END_C_DECLS
