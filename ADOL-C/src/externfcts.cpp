/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts.cpp
 Revision: $Id$
 Contents: functions and data types for extern (differentiated) functions.
 
 Copyright (c) Andreas Kowarz
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
         
----------------------------------------------------------------------------*/

#include <adolc/externfcts.h>
#include "externfcts_p.h"
#include "taping_p.h"
#include <adolc/adouble.h>
#include "oplate.h"
#include "buffer_temp.h"

#include <cstring>

/****************************************************************************/
/*                                    extern differentiated functions stuff */

#define ADOLC_BUFFER_TYPE \
   Buffer< ext_diff_fct, ADOLC_ext_fct, EDFCTS_BLOCK_SIZE >
static ADOLC_BUFFER_TYPE buffer(init_ext_diff_fct);
static int oldTraceFlag;

ext_diff_fct *reg_ext_fct(ADOLC_ext_fct ext_fct) {
    return buffer.append(ext_fct);
}

int call_ext_fct(ext_diff_fct *edfct,
                 int n, double *xp, adouble *xa,
                 int m, double *yp, adouble *ya)
{
    int i = 0, ret;
    locint numVals;
    double *vals;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (edfct==NULL) fail(ADOLC_EXT_DIFF_NULLPOINTER_STRUCT);

    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        put_op(ext_diff);
        ADOLC_PUT_LOCINT(edfct->index);
        ADOLC_PUT_LOCINT(n);
        ADOLC_PUT_LOCINT(m);
        ADOLC_PUT_LOCINT(xa[i].loc());
        ADOLC_PUT_LOCINT(ya[i].loc());
        ADOLC_PUT_LOCINT(0);               /* keep space for checkpointing index */

        oldTraceFlag=ADOLC_CURRENT_TAPE_INFOS.traceFlag;
        ADOLC_CURRENT_TAPE_INFOS.traceFlag=0;
    } else oldTraceFlag=0;

    numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
    vals = new double[numVals];
    memcpy(vals, ADOLC_GLOBAL_TAPE_VARS.store,
            numVals * sizeof(double));

    for (i=0; i<n; ++i) xp[i]=xa[i].getValue();
    for (i=0; i<m; ++i) yp[i]=ya[i].getValue();

    ret=edfct->function(n, xp, m, yp);

    memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals,
            numVals * sizeof(double));
    delete[] vals;

    /* update taylor buffer if keep != 0 ; possible double counting as in
     * adouble.cpp => correction in taping.c */

    if (oldTraceFlag != 0) {
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += n;
        ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += m;
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
            for (i=0; i<n; ++i) ADOLC_WRITE_SCAYLOR(xa[i].getValue());
            for (i=0; i<m; ++i) ADOLC_WRITE_SCAYLOR(ya[i].getValue());
        }
    }
    /* write back */
    for (i=0; i<n; ++i) xa[i].setValue(xp[i]);
    for (i=0; i<m; ++i) ya[i].setValue(yp[i]);

    ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;

    return ret;
}

ext_diff_fct *get_ext_diff_fct( int index ) {
    return buffer.getElement(index);
}

void init_ext_diff_fct(ext_diff_fct *edfct) {
    char *ptr;

    ptr = (char *)edfct;
    for (unsigned int i = 0; i < sizeof(ext_diff_fct); ++i) ptr[i]=0;
}

/****************************************************************************/
/*                                                               THAT'S ALL */

