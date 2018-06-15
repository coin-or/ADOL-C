/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     LUsolve.cpp
 Revision: $Id$
 Contents: example for
             * Computation of the determinant of a matrix
               by LU-decomposition of the system matrix without pivoting 
             * application of tapedoc to observe taping of
               the new op_codes for the elementary operations
                  
                     y += x1 * x2;
                     y -= x1 * x2;           

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include "LU.h"

void printmat(const char* name, int m, int n, double** M);
int compareJacs(int m, int n, double** jac1, const char* name1, double** jac2,
                const char* name2);

/****************************************************************************/
/*                                                             MAIN PROGRAM */
/*--------------------------------------------------------------------------*/
int main(int argc, char* argv []) {
    int size  = 5;                       // system size
    if (2 == argc) {
       size = atoi(argv[1]);
    }
    if (1 > size) {
      printf("Usage: OMP_NUM_THREADS=NUMTHREADS ./LUsolve [SIZE_SYSTEM]\n");
      return 1;
    }
    printf("System size is %i.\n", size);

  /* variables */
    const int tag   = 1;                       // tape tag
    const int indep = size*size+size;          // # of indeps
    const int depen = size;                    // # of deps

    // passive variables
    double** A = myalloc2(size, size);
    double* a1 = myalloc1(size);
    double* a2 = myalloc1(size);
    double* b = myalloc1(size);
    double* x = myalloc1(size);
    adouble **AA, *AAp, *Abx;                  // active variables
    double *args = myalloc1(indep);            // arguments
    double **jac = myalloc2(depen,indep);      // the Jacobian
    double *laghessvec = myalloc1(indep);      // Hessian-vector product

    int i,j;

    /*------------------------------------------------------------------------*/
    /* Info */
    fprintf(stdout,"LINEAR SYSTEM SOLVING by "
            "LU-DECOMPOSITION (ADOL-C Example)\n\n");


    /*------------------------------------------------------------------------*/
    /* Allocation und initialization of the system matrix */
    AA  = new adouble*[size];
    AAp = new adouble[size*size];
    for (i=0; i<size; i++) {
        AA[i] = AAp;
        AAp += size;
    }
    Abx = new adouble[size];
    for(i=0; i<size; i++) {
        a1[i] = i*0.25;
        a2[i] = i*0.33;
    }
    for(i=0; i<size; i++) {
        for(j=0; j<size; j++)
            A[i][j] = a1[i]*a2[j];
        A[i][i] += i+1;
        b[i] = -i-1;
    }


    /*------------------------------------------------------------------------*/
    /* Taping the computation of the determinant */
    trace_on(tag);
    /* marking indeps */
    for(i=0; i<size; i++)
        for(j=0; j<size; j++)
            AA[i][j] <<= (args[i*size+j] = A[i][j]);
    for(i=0; i<size; i++)
        Abx[i] <<= (args[size*size+i] = b[i]);
    /* LU-factorization and computation of solution */
    LUfact(size,AA);
    LUsolve(size,AA,Abx);
    /* marking deps */
    for (i=0; i<size; i++)
        Abx[i] >>= x[i];
    trace_off();
    fprintf(stdout," x[0] (original):  %16.4E\n",x[0]);


    /*------------------------------------------------------------------------*/
    /* Recomputation  */
    function(tag,depen,indep,args,x);
    fprintf(stdout," x[0] (from tape): %16.4E\n",x[0]);


    /*------------------------------------------------------------------------*/
    /* Computation of Jacobian */
    jacobian(tag,depen,indep,args,jac);
    if (6 > size)
      printmat("Jacobian", depen, indep, jac);

    double **parJ;
    parJ = myalloc2(depen, indep);
    par_jacobian(tag, depen, indep, args, parJ);
    if (6 > size)
      printmat("Par Jacobian", depen, indep, parJ);

    compareJacs(depen, indep, jac, "jac", parJ, "parJac");

    /*------------------------------------------------------------------------*/
    /* Computation of Lagrange-Hessian-vector product */
    lagra_hess_vec(tag,depen,indep,args,args,x,laghessvec);
    fprintf(stdout," Part of Lagrange-Hessian-vector product:\n");
    if (6 > size) {
      for (i=0; i<size; i++) {
          for (j=0; j<size; j++)
              fprintf(stdout," %14.6E",laghessvec[i*size+j]);
          fprintf(stdout,"\n");
      }
    }


    /*------------------------------------------------------------------------*/
    /* Tape-documentation */
    tape_doc(tag,depen,indep,args,x);


    /*------------------------------------------------------------------------*/
    /* Tape statistics */
    size_t tape_stats[STAT_SIZE];
    tapestats(tag,tape_stats);

    fprintf(stdout,"\n    independents            %zu\n",tape_stats[NUM_INDEPENDENTS]);
    fprintf(stdout,"    dependents              %zu\n",tape_stats[NUM_DEPENDENTS]);
    fprintf(stdout,"    operations              %zu\n",tape_stats[NUM_OPERATIONS]);
    fprintf(stdout,"    operations buffer size  %zu\n",tape_stats[OP_BUFFER_SIZE]);
    fprintf(stdout,"    locations buffer size   %zu\n",tape_stats[LOC_BUFFER_SIZE]);
    fprintf(stdout,"    constants buffer size   %zu\n",tape_stats[VAL_BUFFER_SIZE]);
    fprintf(stdout,"    maxlive                 %zu\n",tape_stats[NUM_MAX_LIVES]);
    fprintf(stdout,"    valstack size           %zu\n\n",tape_stats[TAY_STACK_SIZE]);

    /*------------------------------------------------------------------------*/
    /* That's it */
    return 1;
}


/******************************************************************************/
void printmat(const char* name, int m, int n, double** M)
{
  int i, j;

  printf("%s \n",name);
  for(i = 0; i < m ; ++i) {
    printf("\n %d: ",i);
      for( j = 0; j < n ; ++j)
        printf(" %10.4f ", M[i][j]);
  }
  printf("\n");
}

/******************************************************************************/
int compareJacs(int m, int n, double** jac1, const char* name1, double** jac2,
                const char* name2)
{
  double eps = 1.E-10;
  double f;
  int i, j, ret(0);

  printf("\nCompare %s and %s ...\n", name1, name2);
  for(i = 0; i < m ; ++i) {
    for( j = 0; j < n ; ++j) {
      f = fabs(jac1[i][j] - jac2[i][j]);
      if (f > eps) {
        printf("\tunexpected answer: expected[%d][%d]=%.10f vs result=%.10f\n", i, j, jac1[i][j], jac2[i][j]);
        ret = 1;
      }
    }
  }
  if (!ret)
    printf("\t%s and %s are identical within eps=%.14E \n\n", name1, name2, eps);

  return ret;
}
