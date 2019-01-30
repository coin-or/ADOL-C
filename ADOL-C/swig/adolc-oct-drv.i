%ignore function;
%rename (function) oct_function;
%ignore gradient;
%rename (gradient) oct_gradient;
%ignore jacobian;
%rename (jacobian) oct_jacobian;
%ignore large_jacobian;
%rename (large_jacobian) oct_large_jacobian;
%ignore vec_jac;
%rename (vec_jac) oct_vec_jac;
%ignore jac_vec;
%rename (jac_vec) oct_jac_vec;
%ignore hessian;
%rename (hessian) oct_hessian;
%ignore hessian2;
%rename (hessian2) oct_hessian2;
%ignore hess_vec;
%rename (hess_vec) oct_hess_vec;
%ignore hess_mat;
%rename (hess_mat) oct_hess_mat;
%ignore lagra_hess_vec;
%rename (lagra_hess_vec) oct_lagra_hess_vec;
%ignore sparse_jac;
%rename (sparse_jac) oct_sparse_jac;
%ignore sparse_hess;
%rename (sparse_hess) oct_sparse_hess;
%ignore set_param_vec;
%rename (set_param_vec) oct_set_param_vec;
%ignore tapestats;
%rename (tapestats) oct_tapestats;

%apply (double* IN_ARRAY1, int DIM1) 
       {(double* x, int n1),
        (double* x, int n0),
        (double* u, int n2),
        (double* v, int m1),
        (double* u, int m2)}

%apply (double** ARGOUT_ARRAY1, int* DIM1) 
       {(double** y, int* m1),
        (double** g, int* n2),
        (double** v, int* m2),
        (double** w, int* n2),
        (double** w, int* n3)}

%apply (size_t** ARGOUT_ARRAY1, size_t* DIM1) {(size_t** stats, size_t* stsz)}

%apply (double*** ARGOUT_ARRAY2, int* DIM1, int* DIM2) 
       {(double*** J, int* m2, int* n2),
        (double*** H, int* n2, int* n3),
        (double*** W, int* n2, int* q2)}

%apply (double** IN_ARRAY2, int DIM1, int DIM2)
       {(double** V, int n1, int q)}

%inline %{
#ifdef __cplusplus
extern "C" {
#endif

#define CHECKEXCEPT(rc, func) \
    if ((rc) < 0) {                                                     \
        error("An error has been detected in an ADOL-C library"  \
                     "function (%s). It returned the code %d."          \
                     " Look at previous messages"                       \
                     " printed.",                                       \
                     func, (rc));                                       \
    }

#define DO_GET_DIMENSIONS \
        size_t stats[STAT_SIZE]; \
        int n, m; \
        tapestats(t,stats); \
        n = stats[NUM_INDEPENDENTS]; \
        m = stats[NUM_DEPENDENTS];

    void oct_tapestats(short t, size_t** stats, size_t* stsz) {
        *stsz = STAT_SIZE;
        *stats = (size_t*) malloc(STAT_SIZE*sizeof(size_t));
        tapestats(t,*stats);
    }

    void oct_function(short t, double* x, int n1, double** y, int* m1) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes %d x 1",n
                );
            return;
        }
        int ret;
        *m1 = m;
        *y = (double*)malloc((*m1)*sizeof(double));
        ret = function(t,m,n,x,*y);
        CHECKEXCEPT(ret,"function")
    }
    void oct_gradient(short t, double* x, int n1, double** g, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes %d x 1",n
                );
            return;
        }
        int ret;
        *n2 = n;
        *g = (double*)malloc((*n2)*sizeof(double));
        ret = gradient(t,n,x,*g);
        CHECKEXCEPT(ret,"gradient")
    }
    void oct_jac_vec(short t, double* x, int n1, double* u, int n2, double** v, int* m2) {
        DO_GET_DIMENSIONS
        if (n1 != n || n2 != n) {
            error("Array lengths don't match expected dimensions"
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
    void oct_vec_jac(short t, int repeat, double* x, int n1, double* u, int m2, double** w, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n || m2 != m) {
            error("Array lengths don't match expected dimensions"
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
    void oct_jacobian(short t, double* x, int n1, double*** J, int* m2, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *m2 = m;
        *n2 = n;
        *J = myalloc2(m,n);
        int ret;
        ret = jacobian(t,m,n,x,*J);
        CHECKEXCEPT(ret,"jacobian")
    }
    void oct_large_jacobian(short t, int k, double* x, int n1, double* y, int m1, double*** J, int* m2, int* n2) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *m2 = m;
        *n2 = n;
        int ret;
        *J = myalloc2(m,n);
        ret = large_jacobian(t,m,n,k,x,y,*J);
        CHECKEXCEPT(ret,"large_jacobian")
    }

    void oct_hessian(short t, double* x, int n1, double*** H, int* n2, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *n2 = n;
        *n3 = n;
        *H = myalloc2(n,n);
        int ret;
        ret = hessian(t,n,x,*H);
        CHECKEXCEPT(ret,"hessian")
    }        
    void oct_hessian2(short t, double* x, int n1, double*** H, int* n2, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,)",n
                );
            return;
        }
        *n2 = n;
        *n3 = n;
        *H = myalloc2(n,n);
        int ret;
        ret = hessian2(t,n,x,*H);
        CHECKEXCEPT(ret,"hessian2")
    }
    void oct_hess_vec(short t, double* x, int n1, double* u, int n2, double** w, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n || n2 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,)",n,n
                );
            return;
        }
        int ret;
        *n3 = n;
        *w = (double*)malloc((*n3)*sizeof(double));
        ret = hess_vec(t,n,x,u,*w);
    }
    void oct_hess_mat(short t, double* x, int n0, double** V, int n1, int q, double*** W, int* n2, int* q2) {
        DO_GET_DIMENSIONS
        if (n0 != n || n1 != n) {
            error("Array lengths don't match expected dimensions"
                         "\nExpected shapes (%d,), (%d,any)",n,n
                );
            return;
        }
        *n2 = n;
        *q2 = q;
        *W = myalloc2(n,q);
        int ret;
        ret = hess_mat(t,n,q,x,V,*W);
        CHECKEXCEPT(ret,"hess_mat")
    }
    void oct_lagra_hess_vec(short t, double* x, int n1, double* u, int n2, double* v, int m1, double** w, int* n3) {
        DO_GET_DIMENSIONS
        if (n1 != n || n2 != n || m1 != m) {
            error("Array lengths don't match expected dimensions"
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
%clear (size_t** stats, size_t* stsz);
%clear (double*** J, int* m2, int* n2);
%clear (double*** H, int* n2, int* n3);
%clear (double*** W, int* n2, int* q2);
%clear (double** V, int n1, int q1);
