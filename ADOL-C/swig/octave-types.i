%{
#include <octave/config.h>

#include <octave/ov.h>
#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>
#include <octave/Cell.h>
%}

%fragment("Oct_Fragment", "header")
{
    void* copy_mem(void* ptr, size_t len) {
        void *tmp = (void*) malloc(len);
        memcpy(tmp,ptr,len);
        return tmp;
    }
}

%define %octave_typemaps(DATA_TYPE, DIM_TYPE)
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (DATA_TYPE* IN_ARRAY1, DIM_TYPE DIM1)
{
    const octave_value m = $input;
    $1 = (m.is_real_matrix() && ( m.rows() == 1 || m.cols() == 1 ));
}
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) (DATA_TYPE** IN_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2)
{
    const octave_value m = $input;
    $1 = (m.is_real_matrix());
}
%typemap(in) 
        (DATA_TYPE* IN_ARRAY1, DIM_TYPE DIM1) (Matrix m)
{
    m = $input.matrix_value();
    if (m.rows() == 1) $2 = m.cols();
    else if (m.cols() == 1) $2 = m.rows();
    else SWIG_fail;
    $1 = m.fortran_vec();
}
%typemap(in) 
(DATA_TYPE** IN_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2) 
(Matrix m)
{
    m = $input.matrix_value();
    $2 = m.rows();
    $3 = m.cols();
    $1 = myalloc2($2,$3);
    for (int i = 0; i < $2; i++)
        for (int j = 0; j < $3; j++)
            $1[i][j] = m(i,j);
}
%typemap(freearg)
(DATA_TYPE** IN_ARRAY2, DIM_TYPE DIM1, DIM_TYPE DIM2) 
{
    myfree2($1);
}

%typemap(in,numinputs=0) 
        (DATA_TYPE** ARGOUT_ARRAY1, DIM_TYPE* DIM1)
        (DATA_TYPE* data_temp = NULL, DIM_TYPE dim_temp)
{
    $1 = &data_temp;
    $2 = &dim_temp;
}
%typemap(argout)
(DATA_TYPE** ARGOUT_ARRAY1, DIM_TYPE* DIM1)
{
    dim_vector dv = dim_vector::alloc(2);
    dv(0) = *$2;
    dv(1) = 1;
    Matrix mat$argnum(dv);
    for (int i = 0; i < dv(0); i++)
        mat$argnum(i) = (*$1)[i];
    $result->append(octave_value(mat$argnum));
    free(*$1);
}

%typemap(in,numinputs=0) 
        (DATA_TYPE*** ARGOUT_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2)
        (DATA_TYPE** data_temp = NULL, DIM_TYPE row_temp, DIM_TYPE col_temp)
{
    $1 = &data_temp;
    $2 = &row_temp;
    $3 = &col_temp;
}
%typemap(argout)
(DATA_TYPE*** ARGOUT_ARRAY2, DIM_TYPE* DIM1, DIM_TYPE* DIM2)
{
    dim_vector dv = dim_vector::alloc(2);
    dv(0) = *$2;
    dv(1) = *$3;
    Matrix mat$argnum(dv);
    for (int i = 0; i < dv(0); i++)
        for (int j = 0; j < dv(1); j++)
            mat$argnum(i,j) = (*$1)[i][j];
    $result->append(octave_value(mat$argnum));
    myfree2(*$1);
}

%enddef

%octave_typemaps(unsigned int, int)
%octave_typemaps(double, int)
%octave_typemaps(size_t, size_t)
