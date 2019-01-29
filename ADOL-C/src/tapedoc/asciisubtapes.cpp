/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     tapedoc/asciisubtape.cpp
 Revision: $Id$
 Contents: Routine to converts an ascii description of the trace
           to a subtrace of another trace in ADOL-C core or disk

 Copyright (c) Kshitij Kulshreshtha


 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
----------------------------------------------------------------------------*/

#include <adolc/taping.h>
#include <adolc/interfaces.h>
#include "asciisubtapes.hpp"
#include "externfcts_p.h"
#include "taping_p.h"
#include "oplate.h"


short Subtrace::read() {
    short tag = read_ascii_trace_internal(filename.c_str(),tnum, true);
    return tag;
}

int Subtrace::function(int n, double *x, int m, double *y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::zos_forward(tnum,m,n,0,x,y);
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::zos_forward(int n, double *x, int m, double *y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::zos_forward(tnum,m,n,0,x,y);
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::fos_forward(int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::fos_forward(tnum,m,n,0,dp_x,dp_X,dp_y,dp_Y);
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::fov_forward(int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::fov_forward(tnum,m,n,p,dp_x,dpp_X,dp_y,dpp_Y);
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::hos_forward(int n, double *dp_x, int k, double **dpp_X, int m, double *dp_y, double **dpp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::hos_forward_nk(tnum,m,n,k,dp_x,dpp_X,dp_y,dpp_Y);
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::hov_forward(int n, double *dp_x, int k, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::hov_forward(tnum,m,n,k,p,dp_x,dppp_X,dp_y,dppp_Y);
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::fos_reverse(int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::zos_forward(tnum,m,n,1,dp_x,dp_y);
    if (ret >= 0) {
        MINDEC(ret,::fos_reverse(tnum,m,n,dp_U,dp_Z));
    }
    set_nested_ctx(tnum,0);
    return ret;
}

int Subtrace::fov_reverse(int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::zos_forward(tnum,m,n,1,dp_x,dp_y);
    if (ret >= 0) {
        MINDEC(ret,::fov_reverse(tnum,m,n,p,dpp_U,dpp_Z));
    }
    set_nested_ctx(tnum,0);
    return ret;    
}

int Subtrace::hos_ti_reverse(int m, int d, double **dpp_U, int n, double **dpp_Z, double *dp_x, double **dpp_X, double *dp_y, double **dpp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::hos_forward(tnum,m,n,/*degree*/d,/*keep==degree+1*/d+1,dp_x,dpp_X,dp_y,dpp_Y);
    if (ret >= 0) {
        MINDEC(ret,::hos_ti_reverse(tnum,m,n,d,dpp_U,dpp_Z));
    }
    set_nested_ctx(tnum,0);
    return ret;    
}

int Subtrace::hos_ov_reverse(int m, int d, double **dpp_U, int n, int p, double ***dppp_Z, double *dp_x, double ***dppp_X, double *dp_y, double ***dppp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::hov_wk_forward(tnum,m,n,/*degree*/d,/*keep==degree+1*/d+1,p,dp_x,dppp_X,dp_y,dppp_Y);
    if (ret >= 0) {
        MINDEC(ret,::hos_ov_reverse(tnum,m,n,d,p,dpp_U,dppp_Z));
    }
    set_nested_ctx(tnum,0);
    return ret;    
}

int Subtrace::hov_ti_reverse(int m, int d, int p, double ***dppp_U, int n, double ***dppp_Z, double *dp_x, double **dpp_X, double *dp_y, double **dpp_Y) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::hos_forward(tnum,m,n,/*degree*/d,/*keep==degree+1*/d+1,dp_x,dpp_X,dp_y,dpp_Y);
    if (ret >= 0) {
        MINDEC(ret,::hov_ti_reverse(tnum,m,n,d,p,dppp_U,dppp_Z,NULL));
    }
    set_nested_ctx(tnum,0);
    return ret;    
}

int Subtrace::indopro_forward_tight(int n, double *dp_x, int m, unsigned int **ind_dom) {
    int ret;
    set_nested_ctx(tnum,1);
    ret = ::indopro_forward_tight(tnum,m,n,dp_x,ind_dom);
    set_nested_ctx(tnum,0);
    return ret;
}

void Subtrace::dummycall(locint xstart, locint xnum, locint ystart, locint ynum) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    put_op(ext_diff);
    ADOLC_PUT_LOCINT(edf->index);
    ADOLC_PUT_LOCINT(xnum);
    ADOLC_PUT_LOCINT(ynum);
    ADOLC_PUT_LOCINT(xstart);
    ADOLC_PUT_LOCINT(ystart);
    ADOLC_PUT_LOCINT(0);
    update_ext_fct_memory(edf, xnum, ynum);
    if (edf->dp_x_changes) ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += xnum;
    if (edf->dp_y_priorRequired) ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += ynum;
    ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index = edf->index;
}
