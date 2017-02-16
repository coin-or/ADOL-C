/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc-r.i
 Revision: $Id$
 Contents: Provides all C/C++ interfaces of ADOL-C.
           NOTICE: ALL C/C++ headers will be included DEPENDING ON 
           whether the source code is plain C or C/C++ code. 
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

%module adolc
%{
#include <adolc/adolc.h>
%}

%feature("novaluewrapper") adub;
%feature("novaluewrapper") pdouble;
%feature("novaluewrapper") adubref;

%rename(edf_func) ext_diff_fct::function;
%rename(edf2_func) ext_diff_fct_v2::function;
%rename(cpi_func) CpInfos::function;
%rename(eval_func) function;

%rename(declareIndependent) *::operator<<=;
%rename(declareDependent) *::operator>>=;
%ignore zos_forward_nk;
%ignore fos_forward_nk;
%ignore hos_forward_nk;
%ignore function_;
%ignore gradient_;
%ignore jacobian_;
%ignore large_jacobian_;
%ignore vec_jac_;
%ignore jac_vec_;
%ignore hessian_;
%ignore hess_vec_;
%ignore hessian2_;
%ignore hess_mat_;
%ignore lagra_hess_vec_;
%ignore forodec_;
%ignore accodec_;
%ignore hos_forward_;
%ignore fov_forward_;
%ignore hov_forward_;
%ignore hov_wk_forward_;
%ignore fos_reverse_;
%ignore hos_reverse_;
%ignore hos_ti_reverse_;
%ignore hos_ov_reverse_;
%ignore fov_reverse_;
%ignore hov_reverse_;
%ignore hov_ti_reverse_;
%ignore directional_active_gradient_;
%ignore abs_normal_;
%ignore spread1;
%ignore pack1;
%ignore spread2;
%ignore pack2;
%ignore spread3;
%ignore pack3;
%ignore ext_diff_fct;
%ignore ext_diff_fct_v2;
%ignore CpInfos;

%include "adolc_all.hpp"

%ignore frexp;

%include "adubswigfuncs.hpp"

