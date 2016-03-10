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

%rename (function) npy_function;
%rename (gradient) npy_gradient;
%rename (jacobian) npy_jacobian;
%rename (large_jacobian) npy_large_jacobian;
%rename (vec_jac) npy_vec_jac;
%rename (jac_vec) npy_jac_vec;
%rename (hessian) npy_hessian;
%rename (hessian2) npy_hessian2;
%rename (hess_vec) npy_hess_vec;
%rename (hess_mat) npy_hess_mat;
%rename (lagra_hess_vec) npy_lagra_hess_vec;

%apply (double* INPLACE_ARRAY1, int DIM1) 
       {(double* y, int m1),
        (double* g, int n2),
        (double* v, int m2),
        (double* w, int n2),
        (double* w, int n3)}
%apply (double* IN_ARRAY1, int DIM1) 
       {(double* x, int n1),
        (double* x, int n0),
        (double* u, int n2),
        (double* v, int m1),
        (double* u, int m2)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) 
       {(double* J, int m2, int n2),
        (double* H, int n2, int n3),
        (double* W, int n2, int q2)}
%apply (double* IN_ARRAY2, int DIM1, int DIM2)
       {(double* V, int n1, int q1)}

%exception {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
#ifdef __cplusplus
extern "C" {
#endif
    int npy_function(short t, int m, int n, double* x, int n1, double* y, int m1) {
        if (n1 != n || m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,m
                );
            return 1;
        } else {
            return function(t,m,n,x,y);
        }
    }
    int npy_gradient(short t, int n, double* x, int n1, double* g, int n2) {
        if (n1 != n || n2 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,n
                );
            return 1;
        } else {
            return gradient(t,n,x,g);
        }
    }
    int npy_jacobian(short t, int m, int n, double* x, int n1, double* J, int m2, int n2) {
        if (n1 != n || n2 != n || m2 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d)",n,m,n
                );
            return 1;
        } else {
            char *memory = (char*) malloc( m2 * sizeof(double*) );
            char *tmp = memory;
            double **Jp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Jp,tmp,m2,n2, J);
            ret = jacobian(t,m,n,x,Jp);
            free(memory);
            return ret;
        }
    }
    int npy_large_jacobian(short t, int m, int n, int k, double* x, int n1, double* y, int m1, double* J, int m2, int n2) {
        if (n1 != n || n2 != n || m1 != m || m2 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,), (%d,%d)",n,m,m,n
                );
            return 1;
        } else {
            char *memory = (char*) malloc( m2 * sizeof(double*) );
            char *tmp = memory;
            double **Jp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Jp,tmp,m2,n2, J);
            ret = large_jacobian(t,m,n,k,x,y,Jp);
            free(memory);
            return ret;
        }
    }
    int npy_jac_vec(short t, int m, int n, double* x, int n1, double* u, int n2, double* v, int m2) {
        if (n1 != n || n2 != n || m2 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,), (%d,)",n,n,m
                );
            return 1;
        } else {
            return jac_vec(t,m,n,x,u,v);
        }
    }
    int npy_vec_jac(short t, int m, int n, int repeat, double* x, int n1, double* u, int m2, double* w, int n2) {
        if (n1 != n || n2 != n || m2 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,), (%d,)",n,m,n
                );
            return 1;
        } else {
            return vec_jac(t,m,n,repeat,x,u,w);
        }
    }
    int npy_hessian(short t, int n, double* x, int n1, double* H, int n2, int n3) {
        if (n1 != n || n2 != n || n3 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d)",n,n,n
                );
            return 1;
        } else {
            char *memory = (char*) malloc( n2 * sizeof(double*) );
            char *tmp = memory;
            double **Hp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Hp,tmp,n2,n3, H);
            ret = hessian(t,n,x,Hp);
            free(memory);
            return ret;
        }
    }        
    int npy_hessian2(short t, int n, double* x, int n1, double* H, int n2, int n3) {
        if (n1 != n || n2 != n || n3 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d)",n,n,n
                );
            return 1;
        } else {
            char *memory = (char*) malloc( n2 * sizeof(double*) );
            char *tmp = memory;
            double **Hp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Hp,tmp,n2,n3, H);
            ret = hessian2(t,n,x,Hp);
            free(memory);
            return ret;
        }
    }
    int npy_hess_vec(short t, int n, double* x, int n1, double* u, int n2, double* w, int n3) {
        if (n1 != n || n2 != n || n3 != n) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,), (%d,)",n,n,n
                );
            return 1;
        } else {
            return hess_vec(t,n,x,u,w);
        }
    }
    int npy_hess_mat(short t, int n, int q, double* x, int n0, double* V, int n1, int q1, double* W, int n2, int q2) {
        if (n0 != n || n1 != n || n2 != n || q1 != q || q2 != q) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,%d), (%d,%d)",n,n,q,n,q
                );
            return 1;
        } else {
            char *memory = (char*) malloc( (n1 + n2)  * sizeof(double*) );
            char *tmp = memory;
            double **Vp, **Wp;
            int ret;
            tmp = populate_dpp_with_contigdata(&Vp,tmp,n1,q1, V);
            tmp = populate_dpp_with_contigdata(&Wp,tmp,n2,q2, W);
            ret = hess_mat(t,n,q,x,Vp,Wp);
            free(memory);
            return ret;
        }
    }
    int npy_lagra_hess_vec(short t, int m, int n, double* x, int n1, double* u, int n2, double* v, int m1, double* w, int n3) {
        if (n1 != n || n2 != n || n3 != n || m1 != m) {
            PyErr_Format(PyExc_ValueError,
                         "Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,), (%d,), (%d,)",n,n,m,n
                );
            return 1;
        } else {
            return lagra_hess_vec(t,m,n,x,u,v,w);
        }
    }
#ifdef __cplusplus
}
#endif
%}
%clear (double* y, int m1);
%clear (double* g, int n2);
%clear (double* v, int m2);
%clear (double* w, int n2);
%clear (double* w, int n3);
%clear (double* x, int n1);
%clear (double* x, int n0);
%clear (double* u, int n2);
%clear (double* v, int m1);
%clear (double* u, int m2);
%clear (double* J, int m2, int n2);
%clear (double* H, int n2, int n3);
%clear (double* W, int n2, int q2);
%clear (double* V, int n1, int q1);
%exception ;
