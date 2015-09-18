/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     interfacesf.c
 Revision: $Id$
 Contents: Genuine Fortran callable C Interfaces to ADOL-C forward 
           & reverse calls.
 
 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/

#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include <adolc/fortutils.h>

BEGIN_C_DECLS

/*--------------------------------------------------------------------------*/
fint hos_forward_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fd,
                  fint* fk,
                  fdouble* fbase,
                  fdouble* fx,
                  fdouble* fvalue,
                  fdouble* fy) {
    int rc= -1;
    int tag=*ftag, m=*fm, n=*fn, d=*fd, k=*fk;
    double* base = myalloc1(n);
    double* value = myalloc1(m);
    double** X = myalloc2(n,d);
    double** Y = myalloc2(m,d);
    spread1(n,fbase,base);
    spread2(n,d,fx,X);
    rc= hos_forward(tag,m,n,d,k,base,X,value,Y);
    pack2(m,d,Y,fy);
    pack1(m,value,fvalue);
    myfree2(X);
    myfree2(Y);
    myfree1(base);
    myfree1(value);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint zos_forward_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fk,
                  fdouble* fbase,
                  fdouble* fvalue) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn, k=*fk;
    double* base=myalloc1(n);
    double* value = myalloc1(m);
    spread1(n,fbase,base);
    rc=zos_forward(tag,m,n,k,base,value);
    pack1(m,value,fvalue);
    myfree1(base);
    myfree1(value);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_forward_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fd,
                  fint* fp,
                  fdouble* fbase,
                  fdouble* fx,
                  fdouble* fvalue,
                  fdouble* fy) {
    int rc= -1;
    int tag=*ftag, m=*fm, n=*fn, d=*fd, p=*fp;
    double* base = myalloc1(n);
    double* value = myalloc1(m);
    double*** X = myalloc3(n,p,d);
    double*** Y = myalloc3(m,p,d);
    spread1(n,fbase,base);
    spread3(n,p,d,fx,X);
    rc= hov_forward(tag,m,n,d,p,base,X,value,Y);
    pack3(m,p,d,Y,fy);
    pack1(m,value,fvalue);
    myfree3(X);
    myfree3(Y);
    myfree1(base);
    myfree1(value);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint fov_forward_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fp,
                  fdouble* fbase,
                  fdouble* fx,
                  fdouble* fvalue,
                  fdouble* fy) {
    int rc= -1;
    int tag=*ftag, m=*fm, n=*fn, p=*fp;
    double* base = myalloc1(n);
    double* value = myalloc1(m);
    double** X = myalloc2(n,p);
    double** Y = myalloc2(m,p);
    spread1(n,fbase,base);
    spread2(n,p,fx,X);
    rc= fov_forward(tag,m,n,p,base,X,value,Y);
    pack2(m,p,Y,fy);
    pack1(m,value,fvalue);
    myfree2(X);
    myfree2(Y);
    myfree1(base);
    myfree1(value);
    return rc;
}


/*--------------------------------------------------------------------------*/
fint hos_reverse_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fd,
                  fdouble* fu,
                  fdouble* fz) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn, d=*fd;
    double** Z = myalloc2(n,d+1);
    double* u = myalloc1(m);
    spread1(m,fu,u);
    rc=hos_reverse(tag,m,n,d,u,Z);
    pack2(n,d+1,Z,fz);
    myfree2(Z);
    myfree1(u);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint hos_ti_reverse_(
    fint* ftag,
    fint* fm,
    fint* fn,
    fint* fd,
    fdouble* fu,
    fdouble* fz) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn, d=*fd;
    double** Z = myalloc2(n,d+1);
    double** U = myalloc2(m,d+1);
    spread2(m,d+1,fu,U);
    rc=hos_ti_reverse(tag,m,n,d,U,Z);
    pack2(n,d+1,Z,fz);
    myfree2(Z);
    myfree2(U);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint fos_reverse_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fdouble* fu,
                  fdouble* fz) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn;
    double* u = myalloc1(m);
    double* Z = myalloc1(n);
    spread1(m,fu,u);
    rc=fos_reverse(tag,m,n,u,Z);
    pack1(n,Z,fz);
    myfree1(Z);
    myfree1(u);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_reverse_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fd,
                  fint* fq,
                  fdouble* fu,
                  fdouble* fz) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn, d=*fd, q=*fq;
    double** U = myalloc2(q,m);
    double*** Z = myalloc3(q,n,d+1);
    short ** nop = 0;
    spread2(q,m,fu,U);
    rc=hov_reverse(tag,m,n,d,q,U,Z,nop);
    pack3(q,n,d+1,Z,fz);
    myfree3(Z);
    myfree2(U);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint hov_ti_reverse_(
    fint* ftag,
    fint* fm,
    fint* fn,
    fint* fd,
    fint* fq,
    fdouble* fu,
    fdouble* fz) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn, d=*fd, q=*fq;
    double*** U = myalloc3(q,m,d+1);
    double*** Z = myalloc3(q,n,d+1);
    short ** nop = 0;
    spread3(q,m,d+1,fu,U);
    rc=hov_ti_reverse(tag,m,n,d,q,U,Z,nop);
    pack3(q,n,d+1,Z,fz);
    myfree3(Z);
    myfree3(U);
    return rc;
}

/*--------------------------------------------------------------------------*/
fint fov_reverse_(fint* ftag,
                  fint* fm,
                  fint* fn,
                  fint* fq,
                  fdouble* fu,
                  fdouble* fz) {
    int rc=-1;
    int tag=*ftag, m=*fm, n=*fn, q=*fq;
    double** U = myalloc2(q,m);
    double** Z = myalloc2(q,n);
    spread2(q,m,fu,U);
    rc=fov_reverse(tag,m,n,q,U,Z);
    pack2(q,n,Z,fz);
    myfree2(Z);
    myfree2(U);
    return rc;
}


/****************************************************************************/
/*                                                               THAT'S ALL */

END_C_DECLS
