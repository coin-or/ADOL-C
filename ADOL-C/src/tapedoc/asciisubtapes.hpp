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

#ifndef _ASCIISUBTAPES_HPP_
#define _ASCIISUBTAPES_HPP_

#include <string>

#include <adolc/adouble.h>
#include <adolc/edfclasses.h>
#include <adolc/tapedoc/asciitapes.h>

class Subtrace : public EDFobject {
protected:
    std::string filename;
    short tnum;
public:
    Subtrace(short tag, const std::string& fname) 
        : EDFobject(), tnum(tag), filename(fname) {
    }
    virtual short read();
    virtual int function(int n, double *x, int m, double *y);
    virtual int zos_forward(int n, double *x, int m, double *y);
    virtual int fos_forward(int n, double *dp_x, double *dp_X, int m, double *dp_y, double *dp_Y);
    virtual int fov_forward(int n, double *dp_x, int p, double **dpp_X, int m, double *dp_y, double **dpp_Y);
    virtual int hos_forward(int n, double *dp_x, int k, double **dpp_X, int m, double *dp_y, double **dpp_Y);
    virtual int hov_forward(int n, double *dp_x, int k, int p, double ***dppp_X, int m, double *dp_y, double ***dppp_Y);
    virtual int fos_reverse(int m, double *dp_U, int n, double *dp_Z, double *dp_x, double *dp_y);
    virtual int fov_reverse(int m, int p, double **dpp_U, int n, double **dpp_Z, double *dp_x, double *dp_y);
    virtual int indopro_forward_tight(int n, double *dp_x, int m, unsigned int **ind_dom);
    virtual void dummycall(locint xstart, locint xnum, locint ystart, locint ynum);
};

short read_ascii_trace_internal(const char*const fname, short tag, bool issubroutine=false);

#endif
