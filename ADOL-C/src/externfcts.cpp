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

  // this call sets edf->function and edf->index:
  ext_diff_fct * edf=buffer.append(ext_fct);

  // for sanity make sure everything else is properly nullified 
  edf->zos_forward=0;

  edf->fos_forward=0;
  edf->hos_forward=0;
  edf->fov_forward=0;
  edf->hov_forward=0;

  edf->fos_reverse=0;
  edf->hos_reverse=0;
  edf->fov_reverse=0;
  edf->hov_reverse=0;
  
  edf->dp_x=0; 
  edf->dp_X=0; 
  edf->dpp_X=0;
  edf->dppp_X=0;
  edf->dp_y=0;  
  edf->dp_Y=0;  
  edf->dpp_Y=0; 
  edf->dppp_Y=0;

  edf->dp_U=0;  
  edf->dpp_U=0; 
  edf->dp_Z=0;  
  edf->dpp_Z=0; 
  edf->dppp_Z=0;

  edf->spp_nz=0;

  edf->max_n=0;
  edf->max_m=0;

  edf->nestedAdolc=true;
  edf->dp_x_changes=true;
  edf->dp_y_priorRequired=true;

  return edf;
}

int call_ext_fct(ext_diff_fct *edfct,
                 int n, double *xp, adouble *xa,
                 int m, double *yp, adouble *ya)
{
    int i = 0, ret;
    locint numVals = 0;
    double *vals = NULL;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;

    if (xa[n-1].loc()-xa[0].loc()!=(unsigned)n-1 || ya[m-1].loc()-ya[0].loc()!=(unsigned)m-1) fail(ADOLC_EXT_DIFF_LOCATIONGAP);
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

    if (edfct->nestedAdolc) {
      numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
      vals = new double[numVals];
      memcpy(vals, ADOLC_GLOBAL_TAPE_VARS.store,
             numVals * sizeof(double));
    }

    edfct->max_n=(edfct->max_n<n)?n:edfct->max_n;
    edfct->max_m=(edfct->max_m<m)?m:edfct->max_m;

    /* update taylor buffer if keep != 0 ; possible double counting as in
     * adouble.cpp => correction in taping.c */

    if (oldTraceFlag != 0) {
      if (edfct->dp_x_changes) ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += n;
      if (edfct->dp_y_priorRequired) ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += m;
      if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
        if (edfct->dp_x_changes) for (i=0; i<n; ++i) ADOLC_WRITE_SCAYLOR(xa[i].getValue());
        if (edfct->dp_y_priorRequired) for (i=0; i<m; ++i) ADOLC_WRITE_SCAYLOR(ya[i].getValue());
      }
    }

    for (i=0; i<n; ++i) xp[i]=xa[i].getValue();
    if (edfct->dp_y_priorRequired) for (i=0; i<m; ++i) yp[i]=ya[i].getValue();

    ret=edfct->function(n, xp, m, yp);

    if (edfct->nestedAdolc) {
      memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals,
          numVals * sizeof(double));
      delete[] vals;
    }

    /* write back */
    if (edfct->dp_x_changes) for (i=0; i<n; ++i) xa[i].setValue(xp[i]);
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

