
/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     fixpoint.c
 Revision: $Id$
 Contents: all C functions directly accessing at least one of the four tapes
           (operations, locations, constants, value stack)
 
 Copyright (c) Andreas Kowarz, Sebastian Schlenkrich
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include "taping_p.h"
#include <adolc/adolc.h>
#include <adolc/fixpoint.h>
#include "dvlparms.h"

#include <vector>

using namespace std;


/*--------------------------------------------------------------------------*/

/* F(x,u,y,dim_x,dim_u) */
/* norm(x,dim_x)        */
typedef struct {
    locint     edf_index;
    int        sub_tape_num;
    int      (*double_F)(double*, double* ,double*, int, int);
    int      (*adouble_F)(adouble*, adouble*, adouble*, int, int);
    double   (*norm)(double*, int);
    double   (*norm_deriv)(double*, int);
    double     epsilon;
    double     epsilon_deriv;
    int        N_max;
    int        N_max_deriv;
}
fpi_data;


static vector<fpi_data*> fpi_stack;


static int iteration ( int dim_xu, double *xu, int dim_x, double *x_fix ) {
    int i, k;
    double err;
    fpi_data *current = fpi_stack.back();
    for (i=0; i<dim_x; i++) x_fix[i] = xu[i];
    for (k=1; k<=current->N_max; k++) {
        for (i=0; i<dim_x; i++) xu[i] = x_fix[i];
        (*current->double_F)( xu, xu+dim_x, x_fix, dim_x, dim_xu-dim_x );
        for (i=0; i<dim_x; i++) xu[i] = x_fix[i] - xu[i];
        err = (*current->norm)(xu,dim_x);
        if (err<current->epsilon) return k;
    }
    return -1;
}


static int fp_zos_forward ( int dim_xu, double *xu, int dim_x, double *x_fix ) {
    int i, k;
    double err;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint edf_index = ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index;
    fpi_data *current=0;
    vector<fpi_data*>::iterator fpi_stack_iterator;
    for (fpi_stack_iterator=fpi_stack.begin();
            fpi_stack_iterator!=fpi_stack.end();
            ++fpi_stack_iterator) {
        current = *fpi_stack_iterator;
        if (edf_index==current->edf_index) break;
    }
    if (fpi_stack_iterator==fpi_stack.end()) {
        fprintf(stderr,"ADOL-C Error! No edf found for fixpoint iteration.\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    for (i=0; i<dim_x; i++) x_fix[i] = xu[i];
    for (k=1; k<=current->N_max; k++) {
        for (i=0; i<dim_x; i++) xu[i] = x_fix[i];
        (*current->double_F)( xu, xu+dim_x, x_fix, dim_x, dim_xu-dim_x );
        for (i=0; i<dim_x; i++) xu[i] = x_fix[i] - xu[i];
        err = (*current->norm)(xu,dim_x);
        if (err<current->epsilon) return k;
    }
    return -1;
}

static int fp_fos_forward ( int dim_xu, double *xu, double *xu_dot,
                            int dim_x, double *x_fix, double *x_fix_dot) {
    // Piggy back
    int i, k;
    double err, err_deriv;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint edf_index = ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index;
    fpi_data *current=0;
    vector<fpi_data*>::iterator fpi_stack_iterator;
    for (fpi_stack_iterator=fpi_stack.begin();
            fpi_stack_iterator!=fpi_stack.end();
            ++fpi_stack_iterator) {
        current = *fpi_stack_iterator;
        if (edf_index==current->edf_index) break;
    }
    if (fpi_stack_iterator==fpi_stack.end()) {
        fprintf(stderr,"ADOL-C Error! No edf found for fixpoint iteration.\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    for (k=1; (k<current->N_max_deriv)|(k<current->N_max); k++) {
        for (i=0; i<dim_x; i++) xu[i] = x_fix[i];
        for (i=0; i<dim_x; i++) xu_dot[i] = x_fix_dot[i];
        fos_forward ( current->sub_tape_num, dim_x, dim_xu, 0, xu, xu_dot, x_fix, x_fix_dot);
        for (i=0; i<dim_x; i++)  xu[i] = x_fix[i] - xu[i];
        err = (*current->norm)(xu,dim_x);
        for (i=0; i<dim_x; i++) xu_dot[i] = x_fix_dot[i] -  xu_dot[i];
        err_deriv = (*current->norm_deriv)(xu_dot,dim_x);
        if ((err<current->epsilon)&(err_deriv<current->epsilon_deriv)) {
            return k;
        }
    }
    return -1;
}

static int fp_fos_reverse ( int dim_x, double *x_fix_bar, int dim_xu, double *xu_bar, double* /*unused*/, double* /*unused*/) {
    // (d x_fix) / (d x_0) = 0 (!)
    int i, k;
    double err;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    locint edf_index = ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index;
    fpi_data *current=0;
    vector<fpi_data*>::iterator fpi_stack_iterator;
    for (fpi_stack_iterator=fpi_stack.begin();
            fpi_stack_iterator!=fpi_stack.end();
            ++fpi_stack_iterator) {
        current = *fpi_stack_iterator;
        if (edf_index==current->edf_index) break;
    }
    if (fpi_stack_iterator==fpi_stack.end()) {
        fprintf(stderr,"ADOL-C Error! No edf found for fixpoint iteration.\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    double *U = new double[dim_xu];
    double *xi = new double[dim_x];

    for (k=1; k<current->N_max_deriv; k++) {
        for (i=0; i<dim_x; i++) xi[i] = U[i];
        fos_reverse ( current->sub_tape_num, dim_x, dim_xu, xi, U );
        for (i=0; i<dim_x; i++) U[i] += x_fix_bar[i];
        for (i=0; i<dim_x; i++) xi[i] = U[i] - xi[i];
        err = (*current->norm_deriv)(xi,dim_x);
        printf(" fp_fos_reverse: k = %d  err = %e\n",k,err);
        if (err<current->epsilon_deriv) {
            for (i=0; i<dim_xu-dim_x; i++) {
                xu_bar[dim_x+i] += U[dim_x+i];
            }

            delete[] xi;
            delete[] U;
            return k;
        }
    }
    for (i=0; i<dim_xu-dim_x; i++) xu_bar[dim_x+i] += U[dim_x+i];
    delete[] xi;
    delete[] U;
    return -1;
}


int fp_iteration ( int        sub_tape_num,
                   int      (*double_F)(double*, double* ,double*, int, int),
                   int      (*adouble_F)(adouble*, adouble*, adouble*, int, int),
                   double   (*norm)(double*, int),
                   double   (*norm_deriv)(double*, int),
                   double     epsilon,
                   double     epsilon_deriv,
                   int        N_max,
                   int        N_max_deriv,
                   adouble   *x_0,
                   adouble   *u,
                   adouble   *x_fix,
                   int        dim_x,
                   int        dim_u ) {
    int i, k;
    double dummy;
    // add new fp information
    fpi_data *data = new fpi_data;
    data->sub_tape_num = sub_tape_num;
    data->double_F     = double_F;
    data->adouble_F    = adouble_F;
    data->norm         = norm;
    data->norm_deriv   = norm_deriv;
    data->epsilon      = epsilon;
    data->epsilon_deriv = epsilon_deriv;
    data->N_max        = N_max;
    data->N_max_deriv  = N_max_deriv;
    fpi_stack.push_back(data);

    // declare extern differentiated function and data
    ext_diff_fct *edf_iteration = reg_ext_fct(&iteration);
    data->edf_index = edf_iteration->index;
    edf_iteration->zos_forward = &fp_zos_forward;
    edf_iteration->fos_forward = &fp_fos_forward;
    edf_iteration->fos_reverse = &fp_fos_reverse;

    // put x and u together
    adouble   *xu      = new adouble[dim_x+dim_u];
    for (i=0; i<dim_x; i++) xu[i] = x_0[i];
    for (i=0; i<dim_u; i++) xu[dim_x+i] = u[i];

    k = call_ext_fct ( edf_iteration, dim_x+dim_u, xu,
                       dim_x, x_fix );

    // tape near solution
    trace_on(sub_tape_num,1);
    for (i=0; i<dim_x; i++) xu[i] <<= x_fix[i].getValue();
    for (i=0; i<dim_u; i++) xu[dim_x+i] <<= u[i].getValue();
    adouble_F(xu, xu+dim_x, x_fix, dim_x, dim_u);
    for (i=0; i<dim_x; i++) x_fix[i] >>= dummy;
    trace_off();

    delete[] xu;
    return k;
}
