/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     param.cpp
 Revision: $Id$
 Contents: class for parameter dependent functions
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/adouble.h>

#include "taping_p.h"

#include <limits>

pdouble::pdouble(double pval) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    _val = pval;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        _idx = ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc();
        ADOLC_GLOBAL_TAPE_VARS.pStore[_idx] = _val;
    } else {
        _idx = std::numeric_limits<locint>::max();
    }
}

pdouble::pdouble(locint idx) {
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    
    if (idx < ADOLC_GLOBAL_TAPE_VARS.numparam) {
        _val = ADOLC_GLOBAL_TAPE_VARS.pStore[idx];
        _idx = idx;
    } else {
        fprintf(DIAG_OUT, "ADOL-C error: Parameter index %d out of bounds, "
                "# existing parameters = %d\n", idx, 
                ADOLC_GLOBAL_TAPE_VARS.numparam);
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
}

pdouble mkparam(double pval) {
    locint _idx;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        _idx = ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc();
        ADOLC_GLOBAL_TAPE_VARS.pStore[_idx] = pval;
    } else {
        return pval;
    }
    return _idx;
}

pdouble getparam(locint index) {
    return index;
}

locint mkparam_idx(double pval) {
    locint _idx;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        _idx = ADOLC_GLOBAL_TAPE_VARS.paramStoreMgrPtr->next_loc();
        ADOLC_GLOBAL_TAPE_VARS.pStore[_idx] = pval;
    } else {
        fprintf(DIAG_OUT, "ADOL-C error: cannot define indexed parameter "
                "while tracing is turned off!\n");
        adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }
    return _idx;
}

