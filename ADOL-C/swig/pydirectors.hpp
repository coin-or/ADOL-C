/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     pydirectors.hpp
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions in python using swig. This does not compile by itself
           without python, numpy and swig declarations being included

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef _ADOLC_PYDIRECTORS_H_
#define _ADOLC_PYDIRECTORS_H_


class PyEDFwrap;
class PyEDF_iArr_wrap;
class PyEDF_v2_wrap;

class PyEDF {
protected:
    PyEDFwrap* cobj;
public:
    PyEDF();
    virtual ~PyEDF();
    virtual int function(PyObject* args) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int zos_forward(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fos_forward(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fov_forward(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fos_reverse(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fov_reverse(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int call(int n, adouble *xa, int m, adouble *ya);
    virtual int call(int n, advector& x, int m, advector& y);
};

class PyEDF_iArr {
protected:
    PyEDF_iArr_wrap* cobj;
public:
    PyEDF_iArr();
    virtual ~PyEDF_iArr();
    virtual int function(PyObject* args) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int zos_forward(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fos_forward(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fov_forward(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fos_reverse(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int fov_reverse(PyObject* arg) {
        throw FatalError(-1,"Not Implemented", __func__,__FILE__,__LINE__);
    }
    virtual int call(int iArrLen, int* iArr, int n, adouble *xa, int m, adouble *ya);
    virtual int call(int iArrLen, int* iArr, int n, advector& x, int m, advector& y);
    virtual int call(int iArrLen, PyObject* pyarr, int n, advector& x, int m, advector& y);
};

#endif
