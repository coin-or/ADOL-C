/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc-numpy-drv.i
 Revision: $Id$
 Contents: Provides all NumPY driver interfaces of ADOL-C.
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

%pythoncode %{
def as_adouble(arg):
    import numpy as np
    if np.isscalar(arg):
        return adouble(arg)
    elif isinstance(arg,badouble):
        return adouble(arg)
    else:
        arg = np.ascontiguousarray(arg,dtype=np.float64)
        shp = np.shape(arg)
        adata = np.array([adouble(val) for val in arg.flat])
        ret = adata.reshape(shp)
        return ret
%}

%ignore function;
%rename (function) npy_function;
%ignore gradient;
%rename (gradient) npy_gradient;
%ignore jacobian;
%rename (jacobian) npy_jacobian;
%ignore large_jacobian;
%rename (large_jacobian) npy_large_jacobian;
%ignore vec_jac;
%rename (vec_jac) npy_vec_jac;
%ignore jac_vec;
%rename (jac_vec) npy_jac_vec;
%ignore hessian;
%rename (hessian) npy_hessian;
%ignore hessian2;
%rename (hessian2) npy_hessian2;
%ignore hess_vec;
%rename (hess_vec) npy_hess_vec;
%ignore hess_mat;
%rename (hess_mat) npy_hess_mat;
%ignore lagra_hess_vec;
%rename (lagra_hess_vec) npy_lagra_hess_vec;
%ignore sparse_jac;
%rename (sparse_jac) npy_sparse_jac;
%ignore sparse_hess;
%rename (sparse_hess) npy_sparse_hess;
%ignore set_param_vec;
%rename (set_param_vec) npy_set_param_vec;
%ignore directional_active_gradient;
%rename (directional_active_gradient) npy_dir_act_grad;
%ignore abs_normal;
%rename (abs_normal) npy_abs_normal;

%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) 
       {(double** y, int* m1),
        (double** g, int* n2),
        (double** v, int* m2),
        (double** w, int* n2),
        (double** w, int* n3)}
%apply (double* IN_ARRAY1, int DIM1) 
       {(double* x, int n1),
        (double* x, int n0),
        (double* u, int n2),
        (double* v, int m1),
        (double* u, int m2)}
%apply (double** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) 
       {(double** J, int* m2, int* n2),
        (double** H, int* n2, int* n3),
        (double** W, int* n2, int* q2),
        (double** Yy, int* p1, int* q1),
        (double** Jj, int* p2, int* q2),
        (double** Zz, int* p3, int* q3),
        (double** Ll, int* p4, int* q4)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2)
       {(double* V, int n1, int q)}

%apply (int* IN_ARRAY1, int DIM1) 
       {(int* options, int nopt)};
%apply (double** ARGOUTVIEWM_ARRAY1, int* DIM1) 
       {(double** values, int* nnz3)};
%apply (unsigned int** ARGOUTVIEWM_ARRAY1, int* DIM1)
       {(unsigned int** rind, int* nnz1),
        (unsigned int** cind, int* nnz2)};
%apply (short** ARGOUTVIEWM_ARRAY1, int* DIM1) 
       {(short** sigma, int* nn3)};

%inline %{
#ifdef __cplusplus
extern "C" {
#endif
    void npy_function(short t, double* x, int n1, double** y, int* m1) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        int ret;
        *m1 = m;
        *y = (double*)malloc((*m1)*sizeof(double));
        ret = function(t,m,n,x,*y);
        CHECKEXCEPT(ret,"function")
    }
    void npy_gradient(short t, double* x, int n1, double** g, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        int ret;
        *n2 = n;
        *g = (double*)malloc((*n2)*sizeof(double));
        ret = gradient(t,n,x,*g);
        CHECKEXCEPT(ret,"gradient")
    }
    void npy_jacobian(short t, double* x, int n1, double** J, int* m2, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *m2 = m;
        *n2 = n;
        *J = (double*) malloc((*m2) * (*n2) * sizeof(double));
        char *memory = (char*) malloc( (*m2) * sizeof(double*) );
        char *tmp = memory;
        double **Jp;
        int ret;
        tmp = populate_dpp_with_contigdata(&Jp,tmp,*m2,*n2, *J);
        ret = jacobian(t,m,n,x,Jp);
        free(memory);
        CHECKEXCEPT(ret,"jacobian")
    }
    void npy_large_jacobian(short t, int k, double* x, int n1, double* y, int m1, double** J, int* m2, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *m2 = m;
        *n2 = n;
        *J = (double*) malloc((*m2) * (*n2) * sizeof(double));
        char *memory = (char*) malloc( (*m2) * sizeof(double*) );
        char *tmp = memory;
        double **Jp;
        int ret;
        tmp = populate_dpp_with_contigdata(&Jp,tmp,*m2,*n2, *J);
        ret = large_jacobian(t,m,n,k,x,y,Jp);
        free(memory);
        CHECKEXCEPT(ret,"large_jacobian")
    }
    void npy_jac_vec(short t, double* x, int n1, double* u, int n2, double** v, int* m2) {
        DO_GET_DIMENSIONS
        if (n1 != n || n2 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,n
                );
            return;
        }
        int ret;
        *m2 = m;
        *v = (double*)malloc((*m2)*sizeof(double));
        ret = jac_vec(t,m,n,x,u,*v);
        CHECKEXCEPT(ret,"jac_vec")
    }
    void npy_vec_jac(short t, int repeat, double* x, int n1, double* u, int m2, double** w, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n || m2 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,m
                );
            return;
        }
        int ret;
        *n2 = n;
        *w = (double*) malloc((*n2)*sizeof(double));
        ret = vec_jac(t,m,n,repeat,x,u,*w);
        CHECKEXCEPT(ret,"vec_jac")
    }
    void npy_hessian(short t, double* x, int n1, double** H, int* n2, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *n2 = n;
        *n3 = n;
        *H = (double*) malloc((*n2) * (*n3) * sizeof(double));
        char *memory = (char*) malloc( (*n2) * sizeof(double*) );
        char *tmp = memory;
        double **Hp;
        int ret;
        tmp = populate_dpp_with_contigdata(&Hp,tmp,*n2,*n3, *H);
        ret = hessian(t,n,x,Hp);
        free(memory);
        CHECKEXCEPT(ret,"hessian")
    }        
    void npy_hessian2(short t, double* x, int n1, double** H, int* n2, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *n2 = n;
        *n3 = n;
        *H = (double*) malloc((*n2) * (*n3) * sizeof(double));
        char *memory = (char*) malloc( (*n2) * sizeof(double*) );
        char *tmp = memory;
        double **Hp;
        int ret;
        tmp = populate_dpp_with_contigdata(&Hp,tmp,*n2,*n3, *H);
        ret = hessian2(t,n,x,Hp);
        free(memory);
        CHECKEXCEPT(ret,"hessian2")
    }
    void npy_hess_vec(short t, double* x, int n1, double* u, int n2, double** w, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n || n2 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,n
                );
            return;
        }
        int ret;
        *n3 = n;
        *w = (double*)malloc((*n3)*sizeof(double));
        ret = hess_vec(t,n,x,u,*w);
    }
    void npy_hess_mat(short t, double* x, int n0, double* V, int n1, int q, double** W, int* n2, int* q2) {
        DO_GET_DIMENSIONS
        if (n0 != n || n1 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,any)",n,n
                );
            return;
        }
        *n2 = n;
        *q2 = q;
        *W = (double*) malloc((*n2) * (*q2) * sizeof(double));
        char *memory = (char*) malloc( (n1 + (*n2))  * sizeof(double*) );
        char *tmp = memory;
        double **Vp, **Wp;
        int ret;
        tmp = populate_dpp_with_contigdata(&Vp,tmp,n1,q, V);
        tmp = populate_dpp_with_contigdata(&Wp,tmp,*n2,*q2, *W);
        ret = hess_mat(t,n,q,x,Vp,Wp);
        free(memory);
        CHECKEXCEPT(ret,"hess_mat")
    }
    void npy_lagra_hess_vec(short t, double* x, int n1, double* u, int n2, double* v, int m1, double** w, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n || n2 != n || m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,), (%d,)",n,n,m
                );
            return;
        }
        int ret;
        *n3 = n;
        *w = (double*) malloc((*n3) *sizeof(double));
        ret = lagra_hess_vec(t,m,n,x,u,v,*w);
        CHECKEXCEPT(ret,"lagra_hess_vec")
    }

    int npy_sparse_jac(short t, int m, int n, int repeat, double* x, int n1,
                       int* options, int nopt,
                       unsigned int** rind, int* nnz1, 
                       unsigned int** cind, int* nnz2,
                       double** values, int* nnz3) {
#if defined(SPARSE_DRIVERS)
        int nnz, rc;
        rc = sparse_jac(t,m,n,repeat,x,&nnz,rind,cind,values,options);
        *nnz1 = nnz;
        *nnz2 = nnz;
        *nnz3 = nnz;
        CHECKEXCEPT(rc,"sparse_jac")
        return nnz;
#else
        PyErr_Format(PyExc_NotImplementedError,
                     "sparse_jac() has not been compiled in the ADOL-C library");
        return 0;
#endif
    }

    int npy_sparse_hess(short t, int n, int repeat, double* x, int n1,
                        int* options, int nopt,
                        unsigned int** rind, int* nnz1,
                        unsigned int** cind, int* nnz2,
                        double** values, int* nnz3) {
#if defined(SPARSE_DRIVERS)
        int nnz, rc;
        rc = sparse_hess(t,n,repeat,x,&nnz,rind,cind,values,options);
        *nnz1 = nnz;
        *nnz2 = nnz;
        *nnz3 = nnz;
        CHECKEXCEPT(rc,"sparse_hess")
        return nnz;
#else
        PyErr_Format(PyExc_NotImplementedError,
                     "sparse_hess() has not been compiled in the ADOL-C library");
        return 0;
#endif
    }

    void npy_set_param_vec(short t, int n, double* x, int n0) {
        if (n0 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        set_param_vec(t,n,x);
    }

    void npy_dir_act_grad(short t, double* x, int n0, double* v, int m1, double** g, int* n2, short** sigma, int* nn3) {
        DO_GET_DIMENSIONS
        if (n0 != n || m1 != n ) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,n
                );
            return;
        }
        int ret;
        *n2 = n;
        *g = (double*) malloc( *n2 * sizeof(double));
        *nn3 = get_num_switches(t);
        *sigma = (short*) malloc( *nn3 * sizeof(short));
        ret = directional_active_gradient(t,n,x,v,*g,*sigma);
        CHECKEXCEPT(ret,"directional_active_gradient");
    }

    void npy_abs_normal(short t, double* x, int n0, double** y, int* m1, double** g, int* n2, double** v, int* m2, double** w, int* n3, double** Yy, int* p1, int* q1, double** Jj, int* p2, int* q2, double** Zz, int* p3, int* q3, double** Ll, int* p4, int* q4) {
        DO_GET_DIMENSIONS
        if (n0 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        int ret;
        double **Y, **J, **Z, **L;
        double *z, *cz, *cy;
        char *memory, *tmp;
        int s = get_num_switches(t);
        *m1 = m; *n2 = s; *m2 = s; *n3 = m;
        *p1 = m; *q1 = n;
        *p2 = m; *q2 = s;
        *p3 = s; *q3 = n;
        *p4 = s; *q4 = s;
        *y = (double*) calloc( *m1, sizeof(double));
        *g = (double*) calloc( *n2, sizeof(double));
        *v = (double*) calloc( *m2, sizeof(double));
        *w = (double*) calloc( *n3, sizeof(double));
        *Yy = (double*) calloc( *p1 * *q1, sizeof(double));
        *Jj = (double*) calloc( *p2 * *q2, sizeof(double));
        *Zz = (double*) calloc( *p3 * *q3, sizeof(double));
        *Ll = (double*) calloc( *p4 * *q4, sizeof(double));
        z = *g; cz = *v; cy = *w;
        memory = (char*) malloc( (*p1 + *p2 + *p3 + *p4) * sizeof(double*));
        tmp = memory;
        tmp = populate_dpp_with_contigdata(&Y,tmp,*p1,*q1, *Yy);
        tmp = populate_dpp_with_contigdata(&J,tmp,*p2,*q2, *Jj);
        tmp = populate_dpp_with_contigdata(&Z,tmp,*p3,*q3, *Zz);
        tmp = populate_dpp_with_contigdata(&L,tmp,*p4,*q4, *Ll);
        ret = abs_normal(t,m,n,s,x,*y,z,cz,cy,Y,J,Z,L);
        free(memory);
        CHECKEXCEPT(ret,"abs_normal");
    }

#ifdef __cplusplus
}
#endif
%}
%clear (double** y, int* m1);
%clear (double** g, int* n2);
%clear (double** v, int* m2);
%clear (double** w, int* n2);
%clear (double** w, int* n3);
%clear (double* x, int n1);
%clear (double* x, int n0);
%clear (double* u, int n2);
%clear (double* v, int m1);
%clear (double* u, int m2);
%clear (double** J, int* m2, int* n2);
%clear (double** H, int* n2, int* n3);
%clear (double** W, int* n2, int* q2);
%clear (double* V, int n1, int q1);
%clear (int* options, int nopt);
%clear (double** values, int* nnz3);
%clear (unsigned int** rind, int* nnz1);
%clear (unsigned int** cind, int* nnz2);
%clear (short** sigma, int* nn3);
%clear (double** Yy, int* p1, int* q1);
%clear (double** Jj, int* p2, int* q2);
%clear (double** Zz, int* p3, int* q3);
%clear (double** Ll, int* p4, int* q4);
