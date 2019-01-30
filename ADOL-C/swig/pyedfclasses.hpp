/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     pyedfclasses.h
 Revision: $Id$
 Contents: public functions and data types for extern (differentiated)
           functions in python using swig. This does not compile by itself
           without python, numpy and swig declarations being included

 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#ifndef _ADOLC_PYEDFCLASSES_H_
#define _ADOLC_PYEDFCLASSES_H_

#include <adolc/edfclasses.h>
#include "pydirectors.hpp"

class PyEDFwrap : public EDFobject {
protected:
    PyEDF* pyobj;
public:
    PyEDFwrap() : EDFobject() {}
    virtual ~PyEDFwrap() {}
    void setPyObj(PyEDF* o) { pyobj = o; }
    int function(int n, double *x, int m, double* y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m };
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
#else
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
#endif
        //args = Py_BuildValue("OOOO:PyEDFobject_function",no,xa,mo,ya);
        args = PyTuple_Pack(4,no,xa,mo,ya);
        rc = pyobj->function(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        return rc;
    }
    int zos_forward(int n, double *x, int m, double* y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m };
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
#else
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
#endif
        //args = Py_BuildValue("OOOO:PyEDFobject_function",no,xa,mo,ya);
        args = PyTuple_Pack(4,no,xa,mo,ya);
        rc = pyobj->zos_forward(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        return rc;
    }
    int fos_forward(int n, double *x, double *xp, int m, double *y, double *yp) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m };
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* xpa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)xp);
        PyObject* ypa = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)yp);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *xpaa = (PyArrayObject*) xpa, *ypaa = (PyArrayObject*) ypa;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capxp = PyCapsule_New((void*)(xp), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capyp = PyCapsule_New((void*)(yp), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capxp = PyCObject_FromVoidPtr((void*)(xp), NULL);
        PyObject* capyp = PyCObject_FromVoidPtr((void*)(yp), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(xpaa) = capxp;
        PyArray_BASE(ypaa) = capyp;
#else
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(xpaa,capxp);
        PyArray_SetBaseObject(ypaa,capyp);
#endif
        args = PyTuple_Pack(6,no,xa,xpa,mo,ya,ypa);
        rc = pyobj->fos_forward(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(xpaa);
        Py_DECREF(ypaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        return rc;
    }
    int fov_forward(int n, double *x, int p, double **Xp, int m, double *y, double **Yp) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m };
        npy_intp inp[2] = { n,p }, outp[2] = { m,p };
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* xpa = PyArray_SimpleNewFromData(2,inp,NPY_DOUBLE,(void*)&Xp[0][0]);
        PyObject* ypa = PyArray_SimpleNewFromData(2,outp,NPY_DOUBLE,(void*)&Yp[0][0]);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* po = PyInt_FromLong((long int)p);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *xpaa = (PyArrayObject*) xpa, *ypaa = (PyArrayObject*) ypa;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capxp = PyCapsule_New((void*)(Xp), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capyp = PyCapsule_New((void*)(Yp), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capxp = PyCObject_FromVoidPtr((void*)(Xp), NULL);
        PyObject* capyp = PyCObject_FromVoidPtr((void*)(Yp), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(xpaa) = capxp;
        PyArray_BASE(ypaa) = capyp;
#else
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(xpaa,capxp);
        PyArray_SetBaseObject(ypaa,capyp);
#endif
        args = PyTuple_Pack(7,no,xa,po,xpa,mo,ya,ypa);
        rc = pyobj->fov_forward(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(xpaa);
        Py_DECREF(ypaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(po);
        return rc;
    }
    int fos_reverse(int m, double *U, int n, double *Z, double *x, double *y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m };
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* ua = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)U);
        PyObject* za = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)Z);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *uaa = (PyArrayObject*) ua, *zaa = (PyArrayObject*) za;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capu = PyCapsule_New((void*)(U), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capz = PyCapsule_New((void*)(Z), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capu = PyCObject_FromVoidPtr((void*)(U), NULL);
        PyObject* capz = PyCObject_FromVoidPtr((void*)(Z), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(uaa) = capu;
        PyArray_BASE(zaa) = capz;
#else
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(uaa,capu);
        PyArray_SetBaseObject(zaa,capz);
#endif
        args = PyTuple_Pack(6,mo,ua,no,za,xa,ya);
        rc = pyobj->fos_reverse(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(uaa);
        Py_DECREF(zaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        return rc;
    }
    int fov_reverse(int m, int q, double **U, int n, double **Z, double *x, double *y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m };
        npy_intp inq[2] = { n,q }, outq[2] = { m,q };
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* ua = PyArray_SimpleNewFromData(1,outq,NPY_DOUBLE,(void*)&U[0][0]);
        PyObject* za = PyArray_SimpleNewFromData(1,inq,NPY_DOUBLE,(void*)&Z[0][0]);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* qo = PyInt_FromLong((long int)q);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *uaa = (PyArrayObject*) ua, *zaa = (PyArrayObject*) za;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capu = PyCapsule_New((void*)(U), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capz = PyCapsule_New((void*)(Z), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capu = PyCObject_FromVoidPtr((void*)(U), NULL);
        PyObject* capz = PyCObject_FromVoidPtr((void*)(Z), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(uaa) = capu;
        PyArray_BASE(zaa) = capz;
#else
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(uaa,capu);
        PyArray_SetBaseObject(zaa,capz);
#endif
        args = PyTuple_Pack(7,mo,qo,ua,no,za,xa,ya);
        rc = pyobj->fov_reverse(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(uaa);
        Py_DECREF(zaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(qo);
        return rc;
    }
    int call(int n, adouble *xa, int m, adouble *ya) {
        return EDFobject::call(n,xa,m,ya);
    }
    int call(int n, advector& x, int m, advector& y) {
        return EDFobject::call(n,x,m,y);
    }
};

PyEDF::PyEDF() {
     cobj = new PyEDFwrap();
     cobj->setPyObj(this);
}

PyEDF::~PyEDF() {
    delete cobj;
}

int PyEDF::call(int n, adouble *xa, int m, adouble *ya) {
    return cobj->call(n,xa,m,ya);
}
int PyEDF::call(int n, advector& x, int m, advector& y) {
    return cobj->call(n,x,m,y);
}

class PyEDF_iArr_wrap : public EDFobject_iArr {
protected:
    PyEDF_iArr* pyobj;
public:
    PyEDF_iArr_wrap() : EDFobject_iArr() {}
    virtual ~PyEDF_iArr_wrap() {}
    void setPyObj(PyEDF_iArr* o) { pyobj = o; }
    int function(int iArrLen, int* iArr, int n, double *x, int m, double* y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m }, arrsz[1] =  { iArrLen };
        PyObject* iar = PyArray_SimpleNewFromData(1,arrsz,NPY_INT,(void*)iArr);
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* ial = PyInt_FromLong((long int)iArrLen);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *iara = (PyArrayObject*) iar;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capiar = PyCapsule_New((void*)(iArr), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capiar = PyCObject_FromVoidPtr((void*)(iArr), NULL);
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(iara) = capiar;
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
#else
        PyArray_SetBaseObject(iara,capiar);
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
#endif
        //args = Py_BuildValue("OOOO:PyEDFobject_function",no,xa,mo,ya);
        args = PyTuple_Pack(6,ial,iar,no,xa,mo,ya);
        rc = pyobj->function(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(iara);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(ial);
        return rc;
    }
    int zos_forward(int iArrLen, int* iArr, int n, double *x, int m, double* y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m }, arrsz[1] =  { iArrLen };
        PyObject* iar = PyArray_SimpleNewFromData(1,arrsz,NPY_INT,(void*)iArr);
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* ial = PyInt_FromLong((long int)iArrLen);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *iara = (PyArrayObject*) iar;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capiar = PyCapsule_New((void*)(iArr), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capiar = PyCObject_FromVoidPtr((void*)(iArr), NULL);
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(iara) = capiar;
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
#else
        PyArray_SetBaseObject(iara,capiar);
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
#endif
        //args = Py_BuildValue("OOOO:PyEDFobject_function",no,xa,mo,ya);
        args = PyTuple_Pack(6,ial,iar,no,xa,mo,ya);
        rc = pyobj->zos_forward(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(iara);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(ial);
        return rc;
    }
    int fos_forward(int iArrLen, int* iArr, int n, double *x, double *xp, int m, double *y, double *yp) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m }, arrsz[1] =  { iArrLen };
        PyObject* iar = PyArray_SimpleNewFromData(1,arrsz,NPY_INT,(void*)iArr);
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* xpa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)xp);
        PyObject* ypa = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)yp);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* ial = PyInt_FromLong((long int)iArrLen);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *xpaa = (PyArrayObject*) xpa, *ypaa = (PyArrayObject*) ypa;
        PyArrayObject *iara = (PyArrayObject*) iar;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capiar = PyCapsule_New((void*)(iArr), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capxp = PyCapsule_New((void*)(xp), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capyp = PyCapsule_New((void*)(yp), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capiar = PyCObject_FromVoidPtr((void*)(iArr), NULL);
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capxp = PyCObject_FromVoidPtr((void*)(xp), NULL);
        PyObject* capyp = PyCObject_FromVoidPtr((void*)(yp), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(iara) = capiar;
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(xpaa) = capxp;
        PyArray_BASE(ypaa) = capyp;
#else
        PyArray_SetBaseObject(iara,capiar);
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(xpaa,capxp);
        PyArray_SetBaseObject(ypaa,capyp);
#endif
        args = PyTuple_Pack(8,ial,iar,no,xa,xpa,mo,ya,ypa);
        rc = pyobj->fos_forward(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(iara);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(xpaa);
        Py_DECREF(ypaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(ial);
        return rc;
    }
    int fov_forward(int iArrLen, int* iArr, int n, double *x, int p, double **Xp, int m, double *y, double **Yp) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m }, arrsz[1] =  { iArrLen };
        npy_intp inp[2] = { n,p }, outp[2] = { m,p };
        PyObject* iar = PyArray_SimpleNewFromData(1,arrsz,NPY_INT,(void*)iArr);
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* xpa = PyArray_SimpleNewFromData(2,inp,NPY_DOUBLE,(void*)&Xp[0][0]);
        PyObject* ypa = PyArray_SimpleNewFromData(2,outp,NPY_DOUBLE,(void*)&Yp[0][0]);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* po = PyInt_FromLong((long int)p);
        PyObject* ial = PyInt_FromLong((long int)iArrLen);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *xpaa = (PyArrayObject*) xpa, *ypaa = (PyArrayObject*) ypa;
        PyArrayObject *iara = (PyArrayObject*) iar;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capiar = PyCapsule_New((void*)(iArr), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capxp = PyCapsule_New((void*)(Xp), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capyp = PyCapsule_New((void*)(Yp), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capiar = PyCObject_FromVoidPtr((void*)(iArr), NULL);
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capxp = PyCObject_FromVoidPtr((void*)(Xp), NULL);
        PyObject* capyp = PyCObject_FromVoidPtr((void*)(Yp), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(iara) = capiar;
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(xpaa) = capxp;
        PyArray_BASE(ypaa) = capyp;
#else
        PyArray_SetBaseObject(iara,capiar);
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(xpaa,capxp);
        PyArray_SetBaseObject(ypaa,capyp);
#endif
        args = PyTuple_Pack(9,ial,iar,no,xa,po,xpa,mo,ya,ypa);
        rc = pyobj->fov_forward(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(iara);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(xpaa);
        Py_DECREF(ypaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(po);
        Py_DECREF(ial);
        return rc;
    }
    int fos_reverse(int iArrLen, int* iArr, int m, double *U, int n, double *Z, double *x, double *y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m }, arrsz[1] =  { iArrLen };
        PyObject* iar = PyArray_SimpleNewFromData(1,arrsz,NPY_INT,(void*)iArr);
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* ua = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)U);
        PyObject* za = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)Z);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* ial = PyInt_FromLong((long int)iArrLen);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *uaa = (PyArrayObject*) ua, *zaa = (PyArrayObject*) za;
        PyArrayObject *iara = (PyArrayObject*) iar;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capiar = PyCapsule_New((void*)(iArr), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capu = PyCapsule_New((void*)(U), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capz = PyCapsule_New((void*)(Z), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capiar = PyCObject_FromVoidPtr((void*)(iArr), NULL);
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capu = PyCObject_FromVoidPtr((void*)(U), NULL);
        PyObject* capz = PyCObject_FromVoidPtr((void*)(Z), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(iara) = capiar;
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(uaa) = capu;
        PyArray_BASE(zaa) = capz;
#else
        PyArray_SetBaseObject(iara,capiar);
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(uaa,capu);
        PyArray_SetBaseObject(zaa,capz);
#endif
        args = PyTuple_Pack(8,ial,iar,mo,ua,no,za,xa,ya);
        rc = pyobj->fos_reverse(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(iara);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(uaa);
        Py_DECREF(zaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(ial);
        return rc;
    }
    int fov_reverse(int iArrLen, int* iArr, int m, int q, double **U, int n, double **Z, double *x, double *y) {
        PyObject* args = NULL; 
        int rc;
        npy_intp in[1] = { n }, out[1] = { m }, arrsz[1] =  { iArrLen };
        npy_intp inq[2] = { n,q }, outq[2] = { m,q };
        PyObject* iar = PyArray_SimpleNewFromData(1,arrsz,NPY_INT,(void*)iArr);
        PyObject* xa = PyArray_SimpleNewFromData(1,in,NPY_DOUBLE,(void*)x);
        PyObject* ya = PyArray_SimpleNewFromData(1,out,NPY_DOUBLE,(void*)y);
        PyObject* ua = PyArray_SimpleNewFromData(1,outq,NPY_DOUBLE,(void*)&U[0][0]);
        PyObject* za = PyArray_SimpleNewFromData(1,inq,NPY_DOUBLE,(void*)&Z[0][0]);
        PyObject* no = PyInt_FromLong((long int)n);
        PyObject* mo = PyInt_FromLong((long int)m);
        PyObject* qo = PyInt_FromLong((long int)q);
        PyObject* ial = PyInt_FromLong((long int)iArrLen);
        PyArrayObject *xaa = (PyArrayObject*) xa, *yaa = (PyArrayObject*) ya;
        PyArrayObject *uaa = (PyArrayObject*) ua, *zaa = (PyArrayObject*) za;
        PyArrayObject *iara = (PyArrayObject*) iar;
#ifdef SWIGPY_USE_CAPSULE
        PyObject* capiar = PyCapsule_New((void*)(iArr), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capx = PyCapsule_New((void*)(x), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capy = PyCapsule_New((void*)(y), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capu = PyCapsule_New((void*)(U), SWIGPY_CAPSULE_NAME, NULL);
        PyObject* capz = PyCapsule_New((void*)(Z), SWIGPY_CAPSULE_NAME, NULL);
#else
        PyObject* capiar = PyCObject_FromVoidPtr((void*)(iArr), NULL);
        PyObject* capx = PyCObject_FromVoidPtr((void*)(x), NULL);
        PyObject* capy = PyCObject_FromVoidPtr((void*)(y), NULL);
        PyObject* capu = PyCObject_FromVoidPtr((void*)(U), NULL);
        PyObject* capz = PyCObject_FromVoidPtr((void*)(Z), NULL);
#endif
#if NPY_API_VERSION < 0x00000007
        PyArray_BASE(iara) = capiar;
        PyArray_BASE(xaa) = capx;
        PyArray_BASE(yaa) = capy;
        PyArray_BASE(uaa) = capu;
        PyArray_BASE(zaa) = capz;
#else
        PyArray_SetBaseObject(iara,capiar);
        PyArray_SetBaseObject(xaa,capx);
        PyArray_SetBaseObject(yaa,capy);
        PyArray_SetBaseObject(uaa,capu);
        PyArray_SetBaseObject(zaa,capz);
#endif
        args = PyTuple_Pack(9,ial,iar,mo,qo,ua,no,za,xa,ya);
        rc = pyobj->fov_reverse(args);
        if (PyErr_Occurred()) rc = -1;
        Py_DECREF(args);
        Py_DECREF(iara);
        Py_DECREF(xaa);
        Py_DECREF(yaa);
        Py_DECREF(uaa);
        Py_DECREF(zaa);
        Py_DECREF(no);
        Py_DECREF(mo);
        Py_DECREF(qo);
        Py_DECREF(ial);
        return rc;
    }
    int call(int iArrLen, int* iArr, int n, adouble *xa, int m, adouble *ya) {
        return EDFobject_iArr::call(iArrLen,iArr,n,xa,m,ya);
    }
    int call(int iArrLen, int* iArr, int n, advector& x, int m, advector& y) {
        return EDFobject_iArr::call(iArrLen,iArr,n,x,m,y);
    }
    int call(int iArrLen, PyObject* pyarr, int n, advector& x, int m, advector& y) {
        int rc;
        PyArrayObject *array2 = NULL;
        int is_new_object2 = 0;
        npy_intp size[1] = {
            iArrLen 
        };
        int *iArr = NULL;
        array2 = obj_to_array_contiguous_allow_conversion(pyarr,
                                                          NPY_INT,
                                                          &is_new_object2);
        if (!array2 || !require_dimensions(array2, 1) ||
            !require_size(array2, size, 1)) SWIG_fail;
        iArr = (int*) array_data(array2);
        rc = call(iArrLen,iArr,n,x,m,y);
        if (is_new_object2 && array2) {
            Py_DECREF(array2);
        }
        return rc;
    fail:
        if (is_new_object2 && array2) {
            Py_DECREF(array2);
        }
        return -1;
    }
};

PyEDF_iArr::PyEDF_iArr() {
    cobj = new PyEDF_iArr_wrap();
    cobj->setPyObj(this);
}

PyEDF_iArr::~PyEDF_iArr() {
    delete cobj;
}

int PyEDF_iArr::call(int iArrLen, int* iArr, int n, adouble *xa, int m, adouble *ya) {
    return cobj->call(iArrLen,iArr,n,xa,m,ya);
}
int PyEDF_iArr::call(int iArrLen, int* iArr, int n, advector& x, int m, advector& y) {
    return cobj->call(iArrLen,iArr,n,x,m,y);
}
int PyEDF_iArr::call(int iArrLen, PyObject* pyarr, int n, advector& x, int m, advector& y) {
    int rc;
    PyArrayObject *array2 = NULL;
    int is_new_object2 = 0;
    npy_intp size[1] = {
        iArrLen 
    };
    int *iArr = NULL;
    array2 = obj_to_array_contiguous_allow_conversion(pyarr,
                                                      NPY_INT,
                                                      &is_new_object2);
    if (!array2 || !require_dimensions(array2, 1) ||
        !require_size(array2, size, 1)) SWIG_fail;
    iArr = (int*) array_data(array2);
    rc = cobj->call(iArrLen,iArr,n,x,m,y);
    if (is_new_object2 && array2) {
        Py_DECREF(array2);
    }
    return rc;
  fail:
    if (is_new_object2 && array2) {
        Py_DECREF(array2);
    }
    return -1;
}

#endif
