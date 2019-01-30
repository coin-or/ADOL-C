/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc-numpy-for.i
 Revision: $Id$
 Contents: Provides all NumPY forward interfaces of ADOL-C.
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

%ignore tapestats;
%ignore printTapeStats;
%ignore forward;
%ignore zos_forward;
%rename (zos_forward) npy_zos_forward;
%ignore fos_forward;
%rename (fos_forward) npy_fos_forward;
%ignore hos_forward;
%rename (hos_forward) npy_hos_forward;
%ignore fov_forward;
%rename (fov_forward) npy_fov_forward;
%ignore hov_forward;
%rename (hov_forward) npy_hov_forward;
%ignore hov_wk_forward;
%rename (hov_wk_forward) npy_hov_wk_forward;

%rename (int_forward_safe) npy_int_forward_safe;
%rename (int_forward_tight) npy_int_forward_tight;
%rename (indopro_forward_safe) npy_indopro_forward_safe;
%rename (indopro_forward_tight) npy_indopro_forward_tight;
%rename (nonl_ind_forward_safe) npy_nonl_ind_forward_safe;
%rename (nonl_ind_forward_tight) npy_nonl_ind_forward_tight;


%pythoncode %{
def tapestats(t):
    import numpy as np
    stats = np.empty((STAT_SIZE,), np.uintp)
    npy_tapestats(t,stats)
    return stats

def printTapeStats(t):
    s = tapestats(t)
    print( '*** TAPE STATS (tape ', t, ') **********')
    print( 'NUM_INDEPENDENTS = ', s[NUM_INDEPENDENTS]) #                          /* # of independent variables */
    print( 'NUM_DEPENDENTS   = ', s[NUM_DEPENDENTS]) #                              /* # of dependent variables */
    print( 'NUM_MAX_LIVES    = ', s[NUM_MAX_LIVES]) #                                /* max # of live variables */
    print( 'TAY_STACK_SIZE   = ', s[TAY_STACK_SIZE]) #               /* # of values in the taylor (value) stack */
    print( 'OP_BUFFER_SIZE   = ', s[OP_BUFFER_SIZE]) #   /* # of operations per buffer == OBUFSIZE (usrparms.h) */
    print( 'NUM_OPERATIONS   = ', s[NUM_OPERATIONS]) #                               /* overall # of operations */
    print( 'OP_FILE_ACCESS   = ', s[OP_FILE_ACCESS]) #                        /* operations file written or not */
    print( 'NUM_LOCATIONS    = ', s[NUM_LOCATIONS]) #                                 /* overall # of locations */
    print( 'LOC_FILE_ACCESS  = ', s[LOC_FILE_ACCESS]) #                        /* locations file written or not */
    print( 'NUM_VALUES       = ', s[NUM_VALUES]) #                                       /* overall # of values */
    print( 'VAL_FILE_ACCESS  = ', s[VAL_FILE_ACCESS]) #                           /* values file written or not */
    print( 'LOC_BUFFER_SIZE  = ', s[LOC_BUFFER_SIZE]) #   /* # of locations per buffer == LBUFSIZE (usrparms.h) */
    print( 'VAL_BUFFER_SIZE  = ', s[VAL_BUFFER_SIZE]) #      /* # of values per buffer == CBUFSIZE (usrparms.h) */
    print( 'TAY_BUFFER_SIZE  = ', s[TAY_BUFFER_SIZE]) #     /* # of taylors per buffer <= TBUFSIZE (usrparms.h) */
    print( 'NUM_EQ_PROD      = ', s[NUM_EQ_PROD]) #                      /* # of eq_*_prod for sparsity pattern */
    print( 'NO_MIN_MAX       = ', s[NO_MIN_MAX]) #  /* no use of min_op, deferred to abs_op for piecewise stuff */
    print( 'NUM_SWITCHES     = ', s[NUM_SWITCHES]) #                   /* # of abs calls that can switch branch */
    print( 'NUM_PARAM        = ', s[NUM_PARAM]) # /* no of parameters (doubles) interchangable without retaping */
    print( '**********************************')

def get_dimensions(t):
    stats = tapestats(t)
    m = stats[NUM_DEPENDENTS]
    n = stats[NUM_INDEPENDENTS]
    return (m,n)

def forward(t,m,n,**kwargs):
    import numpy as np
    d = 0
    x = None
    p = None
    Xp = None
    keep = 0
    if 'd' in kwargs:
        d = int(kwargs['d'])
    if 'x' in kwargs:
        x = kwargs['x']
    if 'keep' in kwargs:
        keep = int(kwargs['keep'])
    if 'ndir' in kwargs:
        p = int(kwargs['ndir'])
        if keep:
            import warnings as w
            w.warn('Argument "keep" ignored because "ndir" given in forward')
        if 'xdot' in kwargs:
            Xp = kwargs['xdot']
        else:
            raise(NotImplementedError('Argument "ndir" given but not "xdot" in forward, both must be provided'))
    if not x:
        raise(NotImplementedError('Argument "x" missing in forward'))
    xx = np.asanyarray(x)
    if xx.ndim == 1 and not p:
        if d != 0:
            raise(ValueError('Wrong "x" dimension or degree "d" in forward'))
        else: 
            return zos_forward(t,m,n,keep,xx)
    elif xx.ndim == 1 and p and Xp:
        if d != 0:
            return npy_forward(t,m,n,d,p,xx,Xp)
        else:
            return npy_forward(t,m,n,p,xx,Xp)
    elif xx.ndim == 2:
        if m == 1:
            return npy_forward_2(t,n,d,keep,xx)
        else:
            return npy_forward_2(t,m,n,d,keep,xx)

%}


%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) 
       {(double** y, int* d2),
        (double** y, int* m1),
        (double** y, int* m2),
        (double** yp, int* m3)};
%apply (double* IN_ARRAY1, int DIM1) 
       {(double* x, int n1),
        (double* x, int n0),
        (double* xp, int n2)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2) 
       {(double* X, int n1, int d1),
        (double* X, int n2, int p2)};
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) 
       {(double** Y, int* m2, int* d2),
        (double** Y, int* m3, int* p3)};
%apply (double* IN_ARRAY3, int DIM1, int DIM2, int DIM3) 
       {(double* X, int n2, int p2, int d2)};
%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3) 
       {(double** Y, int* m3, int* p3, int* d3)};
%apply (int* IN_ARRAY1, int DIM1) 
       {(int* ndim, int n0)};
%apply (unsigned long INPLACE_ARRAY1[ANY]) {(unsigned long stats[STAT_SIZE])};

%inline %{
    void npy_tapestats(short t, unsigned long stats[STAT_SIZE]) {
        tapestats(t,stats);
    }

#define DO_GET_DIMENSIONS \
        size_t stats[STAT_SIZE]; \
        int n, m; \
        tapestats(t,stats); \
        n = stats[NUM_INDEPENDENTS]; \
        m = stats[NUM_DEPENDENTS];

    void npy_forward_2(short t, int m, int n, int d, int keep, double* X, int n1, int d1, double** Y, int* m2, int* d2) {
        if (n1 != n || d1 != d+1) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,%d)" ,n,d
                );
            return;
        } else {
            *m2 = m;
            *d2 = d+1;
            *Y = (double*)malloc((*m2)*(*d2)*sizeof(double));
            char *memory = (char*)malloc( (n1 + (*m2)) * sizeof(double*) );
            char *tmp = memory;
            double **Xp, **Yp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Xp,tmp,n1,d1, X);
            tmp = populate_dpp_with_contigdata(&Yp,tmp,*m2,*d2, *Y);
            ret = forward(t,m,n,d,keep,Xp,Yp);
            free(memory);
            CHECKEXCEPT(ret,"forward")
            return;
        }
    }
    void npy_forward_2(short t, int n, int d, int keep, double* X, int n1, int d1, double** y, int* d2) {
        if (n1 != n || d1 != d+1) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,%d)",n,d+1
                );
            return;
        } else {
            *d2 = d+1;
            *y = (double*) malloc((*d2)*sizeof(double));
            char *memory = (char*)malloc( n1 * sizeof(double*) );
            char *tmp = memory;
            double **Xp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Xp,tmp,n1,d1, X);
            ret = forward(t,1,n,d,keep,Xp,*y);
            free(memory);
            CHECKEXCEPT(ret,"forward")
            return;
        }
    }
    void npy_forward(short t, int m, int n, int d, int p, double* x, int n1, double* X, int n2, int p2, int d2, double** y, int* m2, double** Y, int* m3, int* p3, int* d3) {
        if (n1 != n || n2 != n || d2 != d || p2 != p) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d,%d)",n,n,p,d
                );
            return;
        } else {
            *m2 = m;
            *m3 = m;
            *p3 = p;
            *d3 = d;
            *y = (double*)malloc((*m2)*sizeof(double));
            *Y = (double*)malloc((*m3)*(*p3)*(*d3)*sizeof(double));
            char *memory = (char*) malloc( (n2 + (*m3)) * sizeof(double**) + ((n2*p2) + ((*m3) *(*p3)))*sizeof(double*));
            char *tmp = memory;
            double ***Xp, ***Yp;
            int ret;
            tmp = populate_dppp_with_contigdata(&Xp,tmp,n2,p2,d2, X);
            tmp = populate_dppp_with_contigdata(&Yp,tmp,*m3,*p3,*d3, *Y);
            ret = forward(t,m,n,d,p,x,Xp,*y,Yp);
            free(memory);
            CHECKEXCEPT(ret,"forward")
            return;
        }
    }
    void npy_forward(short t, int m, int n, int p, double* x, int n1, double* X, int n2, int p2, double** y, int* m2, double** Y, int* m3, int* p3) {
        if (n1 != n || n2 != n || p2 != p) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d)",n,n,p
                );
            return;
        } else {
            *m2 = m;
            *m3 = m;
            *p3 = p;            
            *y = (double*)malloc((*m2)*sizeof(double));
            *Y = (double*)malloc((*m3)*(*p3)*sizeof(double));
            char *memory = (char*) malloc( (n2 + (*m3)) * sizeof(double*) );
            char *tmp = memory;
            double **Xp, **Yp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Xp,tmp,n2,p2, X);
            tmp = populate_dpp_with_contigdata(&Yp,tmp,*m3,*p3, *Y);
            ret = forward(t,m,n,p,x,Xp,*y,Yp);
            free(memory);
            CHECKEXCEPT(ret,"forward")
            return;
        }
    }
#ifdef __cplusplus
extern "C" {
#endif
    void npy_zos_forward(short t, int m, int n, int keep, double* x, int n1, double** y, int* m2) {
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        } else {
            int ret;
            *m2 = m;
            *y = (double*)malloc((*m2)*sizeof(double));
            ret = zos_forward(t,m,n,keep,x,*y);
            CHECKEXCEPT(ret,"zos_forward")
            return;
        }
    }
    void npy_fos_forward(short t, int m, int n, int keep, double* x, int n1, double* xp, int n2, double** y, int* m2, double** yp, int* m3) {
        if (n1 != n || n2 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,n
                );
            return;
        } else {
            int ret;
            *m2 = m;
            *m3 = m;
            *y = (double*)malloc((*m2)*sizeof(double));
            *yp = (double*)malloc((*m3)*sizeof(double));
            ret = fos_forward(t,m,n,keep,x,xp,*y,*yp);
            CHECKEXCEPT(ret,"fos_forward")
            return;
        }
    }
    void npy_fov_forward(short t, int m, int n, int p, double* x, int n1, double* X, int n2, int p2, double** y, int* m2, double** Y, int* m3, int* p3) {
        if (n1 != n || n2 != n || p2 != p) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d)",n,n,p
                );
            return;
        } else {
            *m2 = m;
            *m3 = m;
            *p3 = p;
            *y = (double*)malloc((*m2)*sizeof(double));
            *Y = (double*)malloc((*m3)*(*p3)*sizeof(double));
            char *memory = (char*)malloc((n2 + (*m3)) * sizeof(double*));
            char *tmp = memory;
            double **Xp, **Yp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Xp,tmp,n2,p2, X);
            tmp = populate_dpp_with_contigdata(&Yp,tmp,*m3,*p3, *Y);
            ret = fov_forward(t,m,n,p,x,Xp,*y,Yp);
            free(memory);
            CHECKEXCEPT(ret,"fov_forward")
            return;
        }
    }
    void npy_hos_forward(short t, int m, int n, int d, int keep, double* x, int n0, double* X, int n1, int d1, double** y, int* m1, double** Y, int* m2, int* d2) {
        if (n0 != n || n1 != n || d1 != d) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d)",n,n,d
                );
            return;
        } else {
            *m1 = m;
            *m2 = m;
            *d2 = d;
            *y = (double*)malloc((*m1)*sizeof(double));
            *Y = (double*)malloc((*m2)*(*d2)*sizeof(double));
            char *memory = (char*)malloc((n1 + (*m2)) * sizeof(double*));
            char *tmp = memory;
            double **Xp, **Yp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Xp,tmp,n1,d1, X);
            tmp = populate_dpp_with_contigdata(&Yp,tmp,*m2,*d2, *Y);
            ret = hos_forward(t,m,n,d,keep,x,Xp,*y,Yp);
            free(memory);
            CHECKEXCEPT(ret,"hos_forward")
            return;
        }
    }
    void npy_hov_forward(short t, int m, int n, int d, int p, double* x, int n1, double* X, int n2, int p2, int d2, double** y, int* m2, double** Y, int* m3, int* p3, int* d3) {
        if (n1 != n || n2 != n || d2 != d || p2 != p) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d,%d)",n,n,p,d
                );
            return;
        } else {
            *m2 = m;
            *m3 = m;
            *p3 = p;
            *d3 = d;
            *y = (double*)malloc((*m2)*sizeof(double));
            *Y = (double*)malloc((*m3)*(*p3)*(*d3)*sizeof(double));
            char *memory = (char*)malloc((n2 + (*m3)) * sizeof(double**) + (n2*p2 + (*m3)*(*p3)) * sizeof(double*) );
            char *tmp = memory;
            double ***Xp, ***Yp;
            int ret;
            tmp = populate_dppp_with_contigdata(&Xp,tmp,n2,p2,d2, X);
            tmp = populate_dppp_with_contigdata(&Yp,tmp,*m3,*p3,*d3, *Y);
            ret = hov_forward(t,m,n,d,p,x,Xp,*y,Yp);
            free(memory);
            CHECKEXCEPT(ret,"hov_forward")
            return;
        }
    }
    void npy_hov_wk_forward(short t, int m, int n, int d, int p, int keep, double* x, int n1, double* X, int n2, int p2, int d2, double** y, int* m2, double** Y, int* m3, int* p3, int* d3) {
        if (n1 != n || n2 != n || d2 != d || p2 != p) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d,%d)",n,n,p,d
                );
            return;
        } else {
            *m2 = m;
            *m3 = m;
            *p3 = p;
            *d3 = d;
            *y = (double*)malloc((*m2)*sizeof(double));
            *Y = (double*)malloc((*m3)*(*p3)*(*d3)*sizeof(double));
            char *memory = (char*)malloc((n2 + (*m3)) * sizeof(double**) + (n2*p2 + (*m3)*(*p3)) * sizeof(double*) );
            char *tmp = memory;
            double ***Xp, ***Yp;
            int ret;
            tmp = populate_dppp_with_contigdata(&Xp,tmp,n2,p2,d2, X);
            tmp = populate_dppp_with_contigdata(&Yp,tmp,*m3,*p3,*d3, *Y);
            ret = hov_wk_forward(t,m,n,d,p,keep,x,Xp,*y,Yp);
            free(memory);
            CHECKEXCEPT(ret,"hov_wk_forward")
            return;
        }
    }
#ifdef __cplusplus
}
#endif

%}
%clear (double* X, int n1, int d1);
%clear (double* Y, int m2, int d2);
%clear (double* y, int d2);
%clear (double* x, int n0);
%clear (double* x, int n1);
%clear (double* xp, int n2);
%clear (double* y, int m1);
%clear (double* y, int m2);
%clear (double* yp, int m3);
%clear (double* X, int n2, int p2, int d2);
%clear (double* Y, int m3, int p3, int d3);
%clear (double* X, int n2, int p2);
%clear (double* Y, int m3, int p3);
%clear (int* ndim, int n0);
%clear (unsigned long stats[STAT_SIZE]);
