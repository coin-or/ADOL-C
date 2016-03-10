/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc-python.i
 Revision: $Id$
 Contents: Provides all C/C++ interfaces of ADOL-C.
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.  
 
----------------------------------------------------------------------------*/

%module adolc
%{
#define SWIG_FILE_WITH_INIT
#include <adolc/adolc.h>
%}

%include "numpy.i"

%init %{
import_array();
%}

%feature("novaluewrapper") badouble;
%feature("novaluewrapper") adub;
%feature("novaluewrapper") pdouble;
%feature("novaluewrapper") adubref;
%feature("novaluewrapper") adouble;
%feature("novaluewrapper") advector;

%ignore operator<<;
%ignore operator>>;
%ignore operator<=;
%ignore operator<;
%ignore operator>=;
%ignore operator>;
%ignore operator==;
%ignore operator!=;
%ignore pow;
%ignore *::operator++;
%ignore *::operator--;
%ignore *::operator=;
%ignore zos_forward_nk;
%ignore zos_forward_partx;
%ignore fos_forward_nk;
%ignore fos_forward_partx;
%ignore hos_forward_nk;
%ignore hos_forward_partx;
%ignore fov_forward_partx;
%ignore fov_offset_forward;
%ignore hov_forward_partx;
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
%ignore *::operator[](const badouble&);
%ignore *::operator[](size_t);

%include "adolc_all.hpp"

%extend advector {
    adub* __getitem__(const badouble& index) const {
        return (adub*) (*($self))[index];
    }
    adubref* __getitem__(const badouble& index) {
        return (adubref*) (*($self))[index];
    }
    adouble& __getitem__(size_t index) {
        return (*($self))[index];
    }
    const adouble& __getitem__(size_t index) const {
        return (*($self))[index];
    }
    size_t __len__() const {
        return (*($self)).size();
    }
 }

%ignore frexp;
%ignore operator+;
%ignore operator-;
%ignore operator*;
%ignore operator/;

%include "adubswigfuncs.h"

%extend badouble {
    int __ne__ (const badouble& a) const {
        return (*($self)) != a;
    }
    int __eq__ (const badouble& a) const {
        return (*($self)) == a;
    }
    int __le__ (const badouble& a) const {
        return (*($self)) <= a;
    }
    int __ge__ (const badouble& a) const {
        return (*($self)) >= a;
    }
    int __lt__ (const badouble& a) const {
        return (*($self)) < a;
    }
    int __gt__ (const badouble& a) const {
        return (*($self)) > a;
    }
    int __ne__ (double a) const {
        return (*($self)) != a;
    }
    int __eq__ (double a) const {
        return (*($self)) == a;
    }
    int __le__ (double a) const {
        return (*($self)) <= a;
    }
    int __ge__ (double a) const {
        return (*($self)) >= a;
    }
    int __lt__ (double a) const {
        return (*($self)) < a;
    }
    int __gt__ (double a) const {
        return (*($self)) > a;
    }
    adouble __pow__ (const badouble& a) {
        return pow((*($self)),a);
    }
    adub* __pow__(double c) {
        return (adub*) pow((*($self)),c);
    }
    adouble __rpow__(double c) {
        return pow(c,(*($self)));
    }
    adub* __pow__ (const pdouble& a) {
        return (adub*) pow((*($self)),a);
    }
    adub* __neg__() {
        return (adub*)( - (*($self)));
    }
    adub* __add__ (const badouble& a) {
        return (adub*) ((*($self)) + a);
    }
    adub* __add__ (double a) {
        return (adub*) ((*($self)) + a);
    }
    adub* __radd__ (double a) {
        return (adub*) (a + (*($self)));
    }
    adub* __add__ (const pdouble& a) {
        return (adub*) ((*($self)) + a);
    }
    adub* __sub__ (const badouble& a) {
        return (adub*) ((*($self)) - a);
    }
    adub* __sub__ (double a) {
        return (adub*) ((*($self)) - a);
    }
    adub* __rsub__ (double a) {
        return (adub*) (a - (*($self)));
    }
    adub* __sub__ (const pdouble& a) {
        return (adub*) ((*($self)) - a);
    }
    adub* __mul__ (const badouble& a) {
        return (adub*) ((*($self)) * a);
    }
    adub* __mul__ (double a) {
        return (adub*) ((*($self)) * a);
    }
    adub* __rmul__ (double a) {
        return (adub*) (a * (*($self)));
    }
    adub* __mul__ (const pdouble& a) {
        return (adub*) ((*($self)) * a);
    }
    adub* __div__ (const badouble& a) {
        return (adub*) ((*($self)) / a);
    }
    adub* __div__ (double a) {
        return (adub*) ((*($self)) / a);
    }
    adub* __rdiv__ (double a) {
        return (adub*) (a / (*($self)));
    }
    adub* __div__ (const pdouble& a) {
        return (adub*) ((*($self)) / a);
    }    
}

%extend pdouble {
    adub* __neg__() {
        return (adub*)( - (*($self)));
    }
    adub* __add__ (const badouble& a) {
        return (adub*) ((*($self)) + a);
    }
    adub* __add__ (double a) {
        return (adub*) ((*($self)) + a);
    }
    adub* __radd__ (double a) {
        return (adub*) (a + (*($self)));
    }
    adub* __sub__ (const badouble& a) {
        return (adub*) ((*($self)) - a);
    }
    adub* __sub__ (double a) {
        return (adub*) ((*($self)) - a);
    }
    adub* __rsub__ (double a) {
        return (adub*) (a - (*($self)));
    }
    adub* __mul__ (const badouble& a) {
        return (adub*) ((*($self)) * a);
    }
    adub* __mul__ (double a) {
        return (adub*) ((*($self)) * a);
    }
    adub* __rmul__ (double a) {
        return (adub*) (a * (*($self)));
    }
    adub* __div__ (const badouble& a) {
        return (adub*) ((*($self)) / a);
    }
    adub* __div__ (double a) {
        return (adub*) ((*($self)) / a);
    }
    adub* __rdiv__ (double a) {
        return (adub*) (a / (*($self)));
    }
}

%include "adolc-numpy-for.i"
