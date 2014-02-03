/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adalloc.c
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions 
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <adolc/adalloc.h>

#if !defined(ADOLC_NO_MALLOC)
#   define ADOLC_CALLOC(n,m) calloc(n,m)
#else
#   define ADOLC_CALLOC(n,m) rpl_calloc(n,m)
#endif
#if defined(ADOLC_USE_CALLOC)
#  if !defined(ADOLC_NO_MALLOC)
#     define ADOLC_MALLOC(n,m) calloc(n,m)
#  else
#     define ADOLC_MALLOC(n,m) rpl_calloc(n,m)
#  endif
#else
#  if !defined(ADOLC_NO_MALLOC)
#     define ADOLC_MALLOC(n,m) malloc(n*m)
#  else
#     define ADOLC_MALLOC(n,m) rpl_malloc(n*m)
#  endif
#endif

BEGIN_C_DECLS

/****************************************************************************/
/*                                              MEMORY MANAGEMENT UTILITIES */

/*--------------------------------------------------------------------------*/
double* myalloc1(size_t m) {
    double* A = NULL;  
    if (m>0) {
      A=(double*)ADOLC_MALLOC(m,sizeof(double));
      if (A == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myalloc1 cannot allocate %zd bytes\n",
                (size_t)(m*sizeof(double)));
        exit (-1);
      }
    }
    return A;
}

/*--------------------------------------------------------------------------*/
double** myalloc2(size_t m, size_t n) {
    double **A=NULL;
    if (m>0 && n>0)  { 
      int i;
      double *Adum = (double*)ADOLC_MALLOC(m*n,sizeof(double));
      A = (double**)malloc(m*sizeof(double*));
      if (Adum == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myalloc2 cannot allocate %zd bytes\n",
                (size_t)(m*n*sizeof(double)));
        exit (-1);
      }
      if (A == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myalloc2 cannot allocate %zd bytes\n",
                (size_t)(m*sizeof(double*)));
        exit (-1);
      }
      for (i=0; i<m; i++) {
        A[i] = Adum;
        Adum += n;
      }
    }
    return A;
}

/*--------------------------------------------------------------------------*/
double*** myalloc3(size_t m, size_t n, size_t p) { /* This function allocates 3-tensors contiguously */
    double  ***A = NULL;
    if (m>0 && n>0 && p > 0)  { 
      int i,j;
      double *Adum = (double*) ADOLC_MALLOC(m*n*p,sizeof(double));
      double **Apt = (double**)malloc(m*n*sizeof(double*));
      A = (double***)malloc(m*sizeof(double**));
      if (Adum == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myalloc3 cannot allocate %zd bytes\n",
                (size_t)(m*n*p*sizeof(double)));
        exit (-1);
      }
      if (Apt == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myalloc3 cannot allocate %zd bytes\n",
                (size_t)(m*n*sizeof(double*)));
        exit (-1);
      }
      if (A == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myalloc3 cannot allocate %zd bytes\n",
                (size_t)(m*sizeof(double**)));
        exit (-1);
      }
      for (i=0; i<m; i++) {
        A[i] = Apt;
        for (j=0; j<n; j++) {
	  *Apt++ =  Adum;
	  Adum += p;
        }
      }
    }
    return A;
}

/*--------------------------------------------------------------------------*/
void myfree1(double   *A) {
    if (A) free((char*) A);
}

/*--------------------------------------------------------------------------*/
void myfree2(double  **A) {
    if (A && *A) free((char*)*A);
    if (A) free((char*) A);
}

/*--------------------------------------------------------------------------*/
void myfree3(double ***A) {
    if (A && *A && **A) free((char*)**A);
    if (A && *A) free((char*)*A);
    if (A) free((char*) A);
}


/****************************************************************************/
/*                                          SPECIAL IDENTITY REPRESENTATION */

/*--------------------------------------------------------------------------*/
double   **myallocI2(int n) {
    double *Idum = (double*)ADOLC_MALLOC((2*n-1),sizeof(double));
    double   **I = (double**)malloc(n*sizeof(double*));
    int i;
    if (Idum == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myallocI2 cannot allocate %i bytes\n",
                (int)((2*n-1)*sizeof(double)));
        exit (-1);
    }
    if (I == NULL) {
        fprintf(DIAG_OUT,"ADOL-C error: myallocI2 cannot allocate %i bytes\n",
                (int)(n*sizeof(double*)));
        exit (-1);
    }
    Idum += (n - 1);
    I[0] = Idum;
    *Idum = 1.0;
    /* 20020628 olvo n3l: Initialization to 0 */
    for (i=1; i<n; i++)
        *(++Idum)= 0.0;
    Idum-=(n-1);
    for (i=1; i<n; i++) {
        I[i] = --Idum;
        *Idum = 0.0;
    }
    return I;
}

/*--------------------------------------------------------------------------*/
void myfreeI2(int n, double** I) {
    free((char*)I[n-1]);
    free((char*) I);
}

/****************************************************************************/
/*                              INTEGER VARIANT FOR BIT PATTERN PROPAGATION */

/* ------------------------------------------------------------------------- */
unsigned int *myalloc1_uint(int m) {
    unsigned int *A = (unsigned int*)ADOLC_MALLOC(m,sizeof(unsigned int));
    if (A == NULL) {
        fprintf(DIAG_OUT, "ADOL-C error, "__FILE__
                ":%i : \nmyalloc1_ushort cannot allocate %i bytes\n",
                __LINE__, (int)(m*sizeof(unsigned int)));
        exit (-1);
    } /* endif */
    return A;
}


/* ------------------------------------------------------------------------- */
unsigned long int *myalloc1_ulong(int m) {
    unsigned long int *A = (unsigned long int*)ADOLC_CALLOC(m,sizeof(unsigned long int));
    if (A == NULL) {
        fprintf(DIAG_OUT, "ADOL-C error, "__FILE__
                ":%i : \nmyalloc1_ulong cannot allocate %i bytes\n",
                __LINE__, (int)(m*sizeof(unsigned long int)));
        exit (-1);
    } /* endif */
    return A;
}


/* ------------------------------------------------------------------------- */
unsigned long int **myalloc2_ulong(int m,int n) {
    unsigned long int *Adum = (unsigned long int*)ADOLC_CALLOC(m*n,sizeof(unsigned long int));
    unsigned long int **A   = (unsigned long int**)ADOLC_CALLOC(m,sizeof(unsigned long int*));
    int i;
    if (Adum == NULL) {
        fprintf(DIAG_OUT, "ADOL-C error, "__FILE__
                ":%i : \nmyalloc2_ulong cannot allocate %i bytes\n",
                __LINE__, (int)(m*n*sizeof(unsigned long int)));
        exit (-1);
    } /* endif */
    if (A == NULL) {
        fprintf(DIAG_OUT, "ADOL-C error, "__FILE__
                ":%i : \nmyalloc2_ulong cannot allocate %i bytes\n",
                __LINE__, (int)(m*sizeof(unsigned long int*)));
        exit (-1);
    } /* endif */
    for(i=0;i<m;i++) {
        A[i] = Adum;
        Adum += n;
    }
    return A;

    /* To deallocate an array set up by   A = myalloc2_ulong(m,n)   */
    /*    use  free((char*)*A); free((char*)A);  in that order      */

}


/* ------------------------------------------------------------------------ */

void myfree1_uint(unsigned int *A) {
    free((char *)A);
}

/* ------------------------------------------------------------------------ */

void myfree1_ulong(unsigned long int *A) {
    free((char *)A);
}

/* ------------------------------------------------------------------------ */

void myfree2_ulong(unsigned long int **A) {
    free((char *)*A);
    free((char *)A);
}


END_C_DECLS

