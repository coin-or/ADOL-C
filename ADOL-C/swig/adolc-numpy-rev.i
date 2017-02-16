/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc-numpy-rev.i
 Revision: $Id$
 Contents: Provides all NumPY reverse interfaces of ADOL-C.
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

%ignore reverse;
%ignore fos_reverse;
%rename (fos_reverse) npy_fos_reverse;
%ignore hos_reverse;
%rename (hos_reverse) npy_hos_reverse;
%ignore fov_reverse;
%rename (fov_reverse) npy_fov_reverse;
%ignore hov_reverse;
%rename (hov_reverse) npy_hov_reverse;

%apply (double* IN_ARRAY1, int DIM1) 
       {(double* u, int m1),
        (double* u, int q),
        (double* u, int q1)};
%apply (double* IN_ARRAY2, int DIM1, int DIM2)
       {(double* u, int q1, int m1),
        (double* u, int q, int m1)};
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) 
       {(double** Z, int* n2, int* d2),
        (double** Z, int* q2, int* n2)};
%apply (short* INPLACE_ARRAY2, int DIM1, int DIM2)
       {(short* nz, int q3, int n3)};
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) 
       {(double** Z, int* n2)};
%apply (double** ARGOUTVIEWM_ARRAY3, int* DIM1, int* DIM2, int* DIM3)
       {(double** Z, int* q2, int* n2, int* d2)};

%pythoncode %{
def reverse(t,**kwargs):
    import numpy as np
    d = 0
    nz = False
    u = None
    if 'u' in kwargs:
        u = kwargs['u']
    if 'd' in kwargs:
        d = int(kwargs['d'])
    if 'nz' in kwargs:
        nz = kwargs['nz']
    (m,n) = get_dimensions(t)
    if not u:
        if nz:
            nzout = np.empty((m,n),np.short)
            Z = npy_reverse_nz(t,d,nzout)
            return (Z,nzout)
        else:
            return npy_reverse(t,d)
    elif d == 0:
        if isinstance(u, (int, long, float)):
            uu = float(u)
            return npy_reverse(t,uu)
        else:
            uu = np.asanyarray(u)
            if uu.ndim == 1 and m == 1:
                return npy_reverse_v(t,uu)
            elif uu.ndim == 1:
                return npy_reverse(t,uu)
            elif uu.ndim == 2:
                return npy_reverse_v2(t,uu)
            else:
                raise(NotImplementedError('Wrong "u" dimension in reverse'))
    else:
        if isinstance(u, (int, long, float)):
            uu = float(u)
            return npy_reverse(t,d,uu)
        else:
            uu = np.asanyarray(u)
            if uu.ndim == 1 and m == 1:
                if nz:
                    nzout = np.empty((uu.shape[0],n),np.short)
                    Z = npy_reverse_v_nz(t,d,uu,nzout)
                    return (Z,nzout)
                else:
                    return npy_reverse_v(t,d,uu)
            elif uu.ndim == 1:
                return npy_reverse(t,d,uu)
            elif uu.ndim == 2:
                if nz:
                    nzout = np.empty((uu.shape[0],n),np.short)
                    Z = npy_reverse_v2_nz(t,d,uu,nzout)
                    return (Z,nzout)
                else:
                    return npy_reverse_v2(t,d,uu)
            else:
                raise(NotImplementedError('Wrong "u" dimension in reverse'))

%}

%inline %{
    void npy_reverse(short t, double u, double** Z, int* n2) {
        DO_GET_DIMENSIONS
        if (m != 1) {
            PyErr_Format(PyExc_AssertionError,
                         "Expected number of dependents = 1, got %d",m
                );
            return;
        }
        *n2 = n;
        *Z = (double*)malloc((*n2)*sizeof(double));
        int ret;
        ret = reverse(t,1,n,0,u,*Z);
        return;
    }
    void npy_reverse(short t, double* u, int m1, double** Z, int* n2) {
        DO_GET_DIMENSIONS
        if (m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)" ,m
                );
            return;
        } else {
            *n2 = n;
            *Z = (double*)malloc((*n2)*sizeof(double));
            int ret;
            ret = reverse(t,m,n,0,u,*Z);
            return;
        }
    }
    void npy_reverse_v(short t, double* u, int q, double** Z, int* q2, int* n2) {
        DO_GET_DIMENSIONS
        if (m != 1) {
            PyErr_Format(PyExc_AssertionError,
                         "Expected number of dependents = 1, got %d",m
                );
            return;
        }
        *n2 = n;
        *q2 = q;
        *Z = (double*)malloc((*q2)*(*n2)*sizeof(double));
        int ret;
        char *memory = (char*)malloc((*q2)*sizeof(double*));
        char *tmp = memory;
        double **Zp;
        tmp = populate_dpp_with_contigdata(&Zp,tmp,*q2,*n2, *Z);
        ret = reverse(t,1,n,0,q,u,Zp);
        free(memory);
        return;
    }
    void npy_reverse_v2(short t, double* u, int q, int m1, double** Z, int* q2, int* n2) {
        DO_GET_DIMENSIONS
        if (m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,%d)" ,q,m
                );
            return;
        } else {
            *n2 = n;
            *q2 = q;
            *Z = (double*)malloc((*q2)*(*n2)*sizeof(double));
            int ret;
            char *memory = (char*)malloc((q + (*q2))*sizeof(double*));
            char *tmp = memory;
            double **Zp, **Up;
            tmp = populate_dpp_with_contigdata(&Zp,tmp,*q2,*n2, *Z);
            tmp = populate_dpp_with_contigdata(&Up,tmp,q,m1, u);
            ret = reverse(t,m,n,0,q,Up,Zp);
            free(memory);
            return;
        }
    }
    void npy_reverse(short t, int d, double u, double** Z, int* n2, int* d2) {
        DO_GET_DIMENSIONS
        if (m != 1) {
            PyErr_Format(PyExc_AssertionError,
                         "Expected number of dependents = 1, got %d",m
                );
            return;
        }
        *n2 = n;
        *d2 = d+1;
        *Z = (double*)malloc((*n2)*(*d2)*sizeof(double));
        char *memory = (char*)malloc((*n2)*sizeof(double*));
        char *tmp = memory;
        double **Zp;
        int ret;
        tmp = populate_dpp_with_contigdata(&Zp,tmp,*n2,*d2, *Z);
        ret = reverse(t,1,n,d,u,Zp);
        free(memory);
        return;
    }
    void npy_reverse(short t, int d, double* u, int m1, double** Z, int* n2, int* d2) {
        DO_GET_DIMENSIONS
        if (m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)" ,m
                );
            return;
        } else {
            *n2 = n;
            *d2 = d+1;
            *Z = (double*)malloc((*n2)*(*d2)*sizeof(double));
            char *memory = (char*)malloc((*n2)*sizeof(double*));
            char *tmp = memory;
            double **Zp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Zp,tmp,*n2,*d2, *Z);
            ret = reverse(t,m,n,d,u,Zp);
            free(memory);
            return;
        }
    }
    void npy_reverse_v2(short t, int d, double* u, int q, int m1, double** Z, int* q2, int* n2, int* d2) {
        DO_GET_DIMENSIONS
        if (m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,%d)" ,q,m
                );
            return;
        } else {
            *n2 = n;
            *q2 = q;
            *d2 = d+1;
            *Z = (double*)malloc((*q2)*(*n2)*(*d2)*sizeof(double));
            int ret;
            char *memory = (char*)malloc((*q2)*sizeof(double**) + (q + (*q2)*(*n2))*sizeof(double*));
            char *tmp = memory;
            double ***Zp, **Up;
            tmp = populate_dpp_with_contigdata(&Up,tmp,q,m1, u);
            tmp = populate_dppp_with_contigdata(&Zp,tmp,*q2,*n2,*d2, *Z);
            ret = reverse(t,m,n,d,q,Up,Zp);
            free(memory);
            return;
        }
    }
    void npy_reverse_v2_nz(short t, int d, double* u, int q, int m1, double** Z, int* q2, int* n2, int* d2, short* nz, int q3, int n3) {
        DO_GET_DIMENSIONS
        if (m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,%d)" ,q,m
                );
            return;
        } else {
            *n2 = n;
            *q2 = q;
            *d2 = d+1;
            *Z = (double*)malloc((*q2)*(*n2)*(*d2)*sizeof(double));
            int ret;
            char *memory = (char*)malloc((*q2)*sizeof(double**) + (q + (*q2)*(*n2))*sizeof(double*) + q3*sizeof(short*));
            char *tmp = memory;
            double ***Zp, **Up;
            short **nzp;
            tmp = populate_dpp_with_contigdata(&Up,tmp,q,m1, u);
            tmp = populate_dppp_with_contigdata(&Zp,tmp,*q2,*n2,*d2, *Z);
            tmp = populate_dpp_with_contigdata(&nzp,tmp,q3,n3, nz);
            ret = reverse(t,m,n,d,q,Up,Zp,nzp);
            free(memory);
            return;
        }
    }
    void npy_reverse_v(short t, int d, double* u, int q, double** Z, int* q2, int* n2, int* d2) {
        DO_GET_DIMENSIONS
        if (m != 1) {
            PyErr_Format(PyExc_AssertionError,
                         "Expected number of dependents = 1, got %d",m
                );
            return;
        }
        *n2 = n;
        *q2 = q;
        *d2 = d+1;
        *Z = (double*)malloc((*q2)*(*n2)*(*d2)*sizeof(double));
        int ret;
        char *memory = (char*)malloc((*q2)*sizeof(double**) + ((*q2)*(*n2))*sizeof(double*));
        char *tmp = memory;
        double ***Zp;
        tmp = populate_dppp_with_contigdata(&Zp,tmp,*q2,*n2,*d2, *Z);
        ret = reverse(t,1,n,d,q,u,Zp);
        free(memory);
        return;
    }
    void npy_reverse_v_nz(short t, int d, double* u, int q, double** Z, int* q2, int* n2, int* d2, short* nz, int q3, int n3) {
        DO_GET_DIMENSIONS
        if (m != 1) {
            PyErr_Format(PyExc_AssertionError,
                         "Expected number of dependents = 1, got %d",m
                );
            return;
        }
        *n2 = n;
        *q2 = q;
        *d2 = d+1;
        *Z = (double*)malloc((*q2)*(*n2)*(*d2)*sizeof(double));
        int ret;
        char *memory = (char*)malloc((*q2)*sizeof(double**) + ((*q2)*(*n2))*sizeof(double*) + q3*sizeof(short*));
        char *tmp = memory;
        double ***Zp;
        short **nzp;
        tmp = populate_dppp_with_contigdata(&Zp,tmp,*q2,*n2,*d2, *Z);
        tmp = populate_dpp_with_contigdata(&nzp,tmp,q3,n3, nz);
        ret = reverse(t,1,n,d,q,u,Zp,nzp);
        free(memory);
        return;
    }
    void npy_reverse(short t, int d, double** Z, int* q2, int* n2, int* d2) {
        DO_GET_DIMENSIONS
        *n2 = n;
        *q2 = m;
        *d2 = d+1;
        *Z = (double*)malloc((*q2)*(*n2)*(*d2)*sizeof(double));
        int ret;
        char *memory = (char*)malloc((*q2)*sizeof(double**) + ((*q2)*(*n2))*sizeof(double*));
        char *tmp = memory;
        double ***Zp;
        tmp = populate_dppp_with_contigdata(&Zp,tmp,*q2,*n2,*d2, *Z);
        ret = reverse(t,m,n,d,Zp);
        free(memory);
        return;
    }
    void npy_reverse_nz(short t, int d, double** Z, int* q2, int* n2, int* d2, short* nz, int q3, int n3) {
        DO_GET_DIMENSIONS
        *n2 = n;
        *q2 = m;
        *d2 = d+1;
        *Z = (double*)malloc((*q2)*(*n2)*(*d2)*sizeof(double));
        int ret;
        char *memory = (char*)malloc((*q2)*sizeof(double**) + ((*q2)*(*n2))*sizeof(double*)  + q3*sizeof(short*));
        char *tmp = memory;
        double ***Zp;
        short **nzp;
        tmp = populate_dppp_with_contigdata(&Zp,tmp,*q2,*n2,*d2, *Z);
        tmp = populate_dpp_with_contigdata(&nzp,tmp,q3,n3, nz);
        ret = reverse(t,m,n,d,Zp,nzp);
        free(memory);
        return;
    }

#ifdef __cplusplus
extern "C" {
#endif
    void npy_fos_reverse(short t, int m, int n, double* u, int m1, double** Z, int* n2) {
        if (m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)" ,m
                );
            return;
        }
        *n2 = n;
        *Z = (double*) malloc((*n2)*sizeof(double));
        fos_reverse(t,m,n,u,*Z);
        return;
    }

    void npy_fov_reverse(short t, int m, int n, int q, double* u, int q1, int m1, double** Z, int* q2, int* n2) {
        if (currently_nested(t)) {
            if (q1 != m || m1 != q) {
                PyErr_Format(PyExc_ValueError,
                             "Array lengths don't match expected dimensions"
                             "\nExpected shapes (%d,%d)",m,q
                    );
                return;
            }
            *q2 = n;
            *n2 = q;
        } else {
            if (q1 != q || m1 == m) {
                PyErr_Format(PyExc_ValueError,
                             "Array lengths don't match expected dimensions"
                             "\nExpected shapes (%d,%d)", q,m
                );
                return;
            }
            *q2 = q;
            *n2 = n;
        }
        *Z = (double*) malloc((*q2)*(*n2)*sizeof(double));
        char* memory = (char*)malloc(q1 + (*q2)*sizeof(double*));
        char *tmp = memory;
        double **Zp, **Up;
        tmp = populate_dpp_with_contigdata(&Zp,tmp,*q2,*n2, *Z);
        tmp = populate_dpp_with_contigdata(&Up,tmp,q1,m1, u);
        fov_reverse(t,m,n,q,Up,Zp);
        free(memory);
        return;
    }

#ifdef __cplusplus
}
#endif
%}

%clear (double*u, int m1);
%clear (double** Z, int* n2, int* d2);
%clear (double* u, int q1, int m1);
%clear (double* u, int q, int m1);
%clear (double* u, int q1);
%clear (double* u, int q);
%clear (double** Z, int* q2, int* n2, int* d2);
%clear (double** Z, int* q2, int* n2);
%clear (short* nz, int q3, int n3);
%clear (double** Z, int* n2);

