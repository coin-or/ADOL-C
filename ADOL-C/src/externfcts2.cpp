/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     externfcts2.cpp
 Revision: $Id$
 Contents: functions and data types for extern (differentiated) functions.

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include "taping_p.h"
#include <adolc/externfcts.h>
#include "externfcts_p.h"
#include <adolc/adouble.h>
#include <adolc/adalloc.h>
#include "oplate.h"
#include "buffer_temp.h"

#include <cstring>

/****************************************************************************/
/*                                    extern differentiated functions stuff */

#define ADOLC_BUFFER_TYPE \
   Buffer< ext_diff_fct_v2, EDFCTS_BLOCK_SIZE >

static ADOLC_BUFFER_TYPE buffer(edf_zero);

void edf_zero(ext_diff_fct_v2 *edf) {
  // sanity settings
  edf->function=0;
  edf->zos_forward=0;
  edf->fos_forward=0;
  edf->fov_forward=0;
  edf->fos_reverse=0;
  edf->fov_reverse=0;
  edf->x = 0;
  edf->y = 0;
  edf->xp = 0;
  edf->yp = 0;
  edf->Xp = 0;
  edf->Yp = 0;
  edf->up = 0;
  edf->zp = 0;
  edf->Up = 0;
  edf->Zp = 0;
  edf->max_nin = 0;
  edf->max_nout = 0;
  edf->max_insz = 0;
  edf->max_outsz = 0;
  edf->nestedAdolc=false;
  edf->dp_x_changes=true;
  edf->dp_y_priorRequired=true;
  edf->context = NULL;
  if (edf->allmem != NULL)
      free(edf->allmem);
  edf->allmem=NULL;
  edf->user_allocated_mem=0;
}

ext_diff_fct_v2 *reg_ext_fct(ADOLC_ext_fct_v2 *ext_fct) {
    ext_diff_fct_v2 *edf = buffer.append();
    edf->function = ext_fct;
    return edf;
}

static void update_ext_fct_memory(ext_diff_fct_v2 *edfct, int nin, int nout, int *insz, int *outsz) {
    int m_isz=0, m_osz=0;
    int i,j;
    for(i=0;i<nin;i++)
        m_isz=(m_isz<insz[i])?insz[i]:m_isz;
    for(i=0;i<nout;i++)
        m_osz=(m_osz<outsz[i])?outsz[i]:m_osz;
    if (edfct->max_nin<nin || edfct->max_nout<nout || edfct->max_insz<m_isz || edfct->max_outsz<m_osz) {
        char* tmp;
        size_t p = nin*m_isz, q = nout*m_osz;
        size_t totalmem =
            (3*nin*m_isz + 3*nout*m_osz
             // + nin*m_isz*p + nout*m_osz*p
             // + q*nout*m_osz + q*nin*m_isz
            )*sizeof(double)
            + (3*nin + 3*nout + nin*m_isz + nout*m_osz
               + q*nout + q*nin)*sizeof(double*)
            + (nin + nout + 2*q)*sizeof(double**);
        if (edfct->allmem != NULL) free(edfct->allmem);
        edfct->allmem=(char*)malloc(totalmem);
        memset(edfct->allmem,0,totalmem);
        tmp = edfct->allmem;
        tmp = populate_dpp(&edfct->x,tmp,nin,m_isz);
        tmp = populate_dpp(&edfct->y,tmp,nout,m_osz);
        tmp = populate_dpp(&edfct->xp,tmp,nin,m_isz);
        tmp = populate_dpp(&edfct->yp,tmp,nout,m_osz);
        tmp = populate_dpp(&edfct->up,tmp,nout,m_osz);
        tmp = populate_dpp(&edfct->zp,tmp,nin,m_isz);
        tmp = populate_dppp_nodata(&edfct->Xp,tmp,nin,m_isz);
        tmp = populate_dppp_nodata(&edfct->Yp,tmp,nout,m_osz);
        tmp = populate_dppp_nodata(&edfct->Up,tmp,nout,m_osz);
        tmp = populate_dppp_nodata(&edfct->Zp,tmp,nin,m_isz);
    }
    edfct->max_nin=(edfct->max_nin<nin)?nin:edfct->max_nin;
    edfct->max_nout=(edfct->max_nout<nout)?nout:edfct->max_nout;
    edfct->max_insz=(edfct->max_insz<m_isz)?m_isz:edfct->max_insz;
    edfct->max_outsz=(edfct->max_outsz<m_osz)?m_osz:edfct->max_outsz;
}

int call_ext_fct(ext_diff_fct_v2 *edfct,
                 int iArrLen, int* iArr,
                 int nin, int nout,
                 int *insz, adouble **x,
                 int *outsz, adouble **y) {
    int ret;
    int oldTraceFlag;
    int i,j; size_t numVals;
    double *vals;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    if (ADOLC_CURRENT_TAPE_INFOS.traceFlag) {
        put_op_reserve(ext_diff_v2, 2*(nin+nout)+iArrLen);
        ADOLC_PUT_LOCINT(edfct->index);
        ADOLC_PUT_LOCINT(iArrLen);
        for(i=0;i<iArrLen;i++)
            ADOLC_PUT_LOCINT(iArr[i]);
        ADOLC_PUT_LOCINT(iArrLen);
        ADOLC_PUT_LOCINT(nin);
        ADOLC_PUT_LOCINT(nout);
        for (i=0;i<nin;i++) {
            if (x[i][insz[i]-1].loc()-x[i][0].loc() != (unsigned)insz[i]-1) fail(ADOLC_EXT_DIFF_LOCATIONGAP);
            ADOLC_PUT_LOCINT(insz[i]);
            ADOLC_PUT_LOCINT(x[i][0].loc());
        }
        for (i=0;i<nout;i++) {
            if (y[i][outsz[i]-1].loc()-y[i][0].loc() != (unsigned)outsz[i]-1) fail(ADOLC_EXT_DIFF_LOCATIONGAP);
            ADOLC_PUT_LOCINT(outsz[i]);
            ADOLC_PUT_LOCINT(y[i][0].loc());
        }
        ADOLC_PUT_LOCINT(nin);
        ADOLC_PUT_LOCINT(nout);
        oldTraceFlag = ADOLC_CURRENT_TAPE_INFOS.traceFlag;
        ADOLC_CURRENT_TAPE_INFOS.traceFlag = 0;
    } else oldTraceFlag = 0;
    if (edfct->nestedAdolc) {
        numVals = ADOLC_GLOBAL_TAPE_VARS.storeSize;
        vals = new double[numVals];
        memcpy(vals,ADOLC_GLOBAL_TAPE_VARS.store, numVals*sizeof(double));
    }
    if (!edfct->user_allocated_mem)
        update_ext_fct_memory(edfct,nin,nout,insz,outsz);
    if (oldTraceFlag != 0) {
        if (edfct->dp_x_changes)
            for(i=0;i<nin;i++)
                ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += insz[i];
        if (edfct->dp_y_priorRequired)
            for(i=0;i<nout;i++)
                ADOLC_CURRENT_TAPE_INFOS.numTays_Tape += outsz[i];
        if (ADOLC_CURRENT_TAPE_INFOS.keepTaylors) {
            if (edfct->dp_x_changes)
                for(i=0;i<nin;i++)
                    for(j=0;j<insz[i];j++)
                        ADOLC_WRITE_SCAYLOR(x[i][j].getValue());
            if (edfct->dp_y_priorRequired)
                for(i=0;i<nout;i++)
                    for(j=0;j<outsz[i];j++)
                        ADOLC_WRITE_SCAYLOR(y[i][j].getValue());
        }
    }

    for(i=0;i<nin;i++)
        for(j=0;j<insz[i];j++)
            edfct->x[i][j] = x[i][j].getValue();

    if (edfct->dp_y_priorRequired)
        for(i=0;i<nout;i++)
            for(j=0;j<outsz[i];j++)
                edfct->y[i][j] = y[i][j].getValue();

    ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index = edfct->index;
    ret=edfct->function(iArrLen,iArr,nin,nout,insz,edfct->x,outsz,edfct->y,edfct->context);

    if (edfct->nestedAdolc) {
        memcpy(ADOLC_GLOBAL_TAPE_VARS.store, vals, numVals*sizeof(double));
        delete[] vals;
        vals = NULL;
    }
    if (edfct->dp_x_changes)
        for(i=0;i<nin;i++)
            for(j=0;j<insz[i];j++)
                x[i][j].setValue(edfct->x[i][j]);

    for(i=0;i<nout;i++)
        for(j=0;j<outsz[i];j++)
            y[i][j].setValue(edfct->y[i][j]);

    ADOLC_CURRENT_TAPE_INFOS.traceFlag=oldTraceFlag;
    return ret;
}

ext_diff_fct_v2 *get_ext_diff_fct_v2( int index ) {
    return buffer.getElement(index);
}

static int edfoo_v2_wrapper_function(int iArrLen, int *iArr, int nin, int nout, int *insz, double **x, int *outsz, double **y, void* ctx) {
    ext_diff_fct_v2* edf;
    EDFobject_v2* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_v2*>(edf->obj);
    return ebase->function(iArrLen,iArr,nin,nout,insz,x,outsz,y,ctx);
}
static int edfoo_v2_wrapper_zos_forward(int iArrLen, int *iArr, int nin, int nout, int *insz, double **x, int *outsz, double **y, void* ctx) {
    ext_diff_fct_v2* edf;
    EDFobject_v2* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_v2*>(edf->obj);
    return ebase->zos_forward(iArrLen,iArr,nin,nout,insz,x,outsz,y,ctx);
}
static int edfoo_v2_wrapper_fos_forward(int iArrLen, int* iArr, int nin, int nout, int *insz, double **x, double **xp, int *outsz, double **y, double **yp, void *ctx) {
    ext_diff_fct_v2* edf;
    EDFobject_v2* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_v2*>(edf->obj);
    return ebase->fos_forward(iArrLen,iArr,nin,nout,insz,x,xp,outsz,y,yp,ctx);
}
static int edfoo_v2_wrapper_fov_forward(int iArrLen, int* iArr, int nin, int nout, int *insz, double **x, int ndir, double ***Xp, int *outsz, double **y, double ***Yp, void* ctx) {
    ext_diff_fct_v2* edf;
    EDFobject_v2* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_v2*>(edf->obj);
    return ebase->fov_forward(iArrLen,iArr,nin,nout,insz,x,ndir,Xp,outsz,y,Yp,ctx);
}
static int edfoo_v2_wrapper_fos_reverse(int iArrLen, int* iArr, int nout, int nin, int *outsz, double **up, int *insz, double **zp, double **x, double **y, void *ctx) {
    ext_diff_fct_v2* edf;
    EDFobject_v2* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_v2*>(edf->obj);
    return ebase->fos_reverse(iArrLen,iArr,nout,nin,outsz,up,insz,zp,x,y,ctx);
}
static int edfoo_v2_wrapper_fov_reverse(int iArrLen, int* iArr, int nout, int nin, int *outsz, int dir, double ***Up, int *insz, double ***Zp, double **x, double **y, void* ctx) {
    ext_diff_fct_v2* edf;
    EDFobject_v2* ebase;
    ADOLC_OPENMP_THREAD_NUMBER;
    ADOLC_OPENMP_GET_THREAD_NUMBER;
    // figure out which edf
    edf = get_ext_diff_fct_v2(ADOLC_CURRENT_TAPE_INFOS.ext_diff_fct_index);
    ebase = reinterpret_cast<EDFobject_v2*>(edf->obj);
    return ebase->fov_reverse(iArrLen,iArr,nout,nin,outsz,dir,Up,insz,Zp,x,y,ctx);
}

void EDFobject_v2::init_edf(EDFobject_v2* ebase) {
    edf = buffer.append();
    edf->obj = reinterpret_cast<void*>(ebase);
    edf->function = edfoo_v2_wrapper_function;
    edf->zos_forward = edfoo_v2_wrapper_zos_forward;
    edf->fos_forward = edfoo_v2_wrapper_fos_forward;
    edf->fov_forward = edfoo_v2_wrapper_fov_forward;
    edf->fos_reverse = edfoo_v2_wrapper_fos_reverse;
    edf->fov_reverse = edfoo_v2_wrapper_fov_reverse;    
}
