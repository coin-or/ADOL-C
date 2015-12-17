%module adolc
%{
#include <adolc/adolc.h>
%}

%feature("novaluewrapper") adub;
%feature("novaluewrapper") pdouble;
%feature("novaluewrapper") adubref;

%rename(__le__) operator<=;
%rename(__lt__) operator<;
%rename(__ge__) operator>=;
%rename(__gt__) operator>;
%rename(__eq__) operator==;
%rename(__ne__) operator!=;
%rename(__pow__) pow;
%ignore *::operator++;
%ignore *::operator--;
%ignore *::operator=;
%ignore *::operator[];
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
%ignore *::operator[](const badouble&);

%include "adolc_all.h"

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

%rename(__add__) operator+;
%rename(__neg__) operator-(const badouble&);
%rename(__neg__) operator-(const pdouble&);
%rename(__sub__) operator-;
%rename(__mul__) operator*;
%rename(__div__) operator/;
%ignore frexp;

%include "adubswigfuncs.h"

