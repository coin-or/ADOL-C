/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     interfaces_mpi.cpp
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/common.h>
#include <adolc/adolc_mpi.h>
#include <adolc/interfaces_mpi.h>

/****** Differentation functions simple use ****************/

/* zos_forward(process id,procsize, tag, m, n, keep, x[n], y[m])            */
int zos_forward( int id,
                 int size,
                 short tag,
                 int m,
                 int n,
                 int keep,
                 const double* x,
                 double* y
){
    int rc=-3;
    if (id==0)
       rc = zos_forward_mpi(id,size,tag,m,n,keep,x,y);
    else
       rc = zos_forward_mpi(id,size,tag,0,0,keep,NULL,NULL);
    return rc;
}

/* fos_forward(process id,procsize, tag, m, n, keep, x[n], X[n], y[m], Y[m])*/
int fos_forward( int id,
                 int size,
                 short tag,
                 int m,
                 int n,
                 int keep,
                 const double* x,
                 double* a,
                 double* y,
                 double* b
){
    int rc=-3;
    if (id==0)
       rc = fos_forward_mpi(id,size,tag,m,n,keep,x,a,y,b);
    else
       rc = fos_forward_mpi(id,size,tag,0,0,keep,NULL,NULL,NULL,NULL);
    return rc;
}

/* fos_reverse(process id, procsize, tag, m, n, u[m], z[n])     */
int fos_reverse( int id,
                 int size,
                 short tag,
                 int m,
                 int n,
                 double* u,
                 double* z
){
    int rc=-3;
    if (id==0)
       rc = fos_reverse_mpi(id,size,tag,m,n,u,z);
    else
       rc = fos_reverse_mpi(id,size,tag,0,0,NULL,NULL);
    return rc;
}

/* hos_forward(process id,procsize, tag, m, n, d, keep, x[n], X[n][d], y[m], Y[m][d])            */
int hos_forward( int id,
                 int size,
                 short tag,
                 int depen,
                 int indep,
                 int d,
                 int keep,
                 double* basepoints,
                 double** argument,
                 double* valuepoints,
                 double** taylors)
{
    int rc=-3;
    if (id==0)
       rc = hos_forward_mpi(id,size,tag,depen,indep,d,keep,basepoints,argument,valuepoints,taylors);
    else
       rc = hos_forward_mpi(id,size,tag,0,0,d,keep,NULL,NULL,NULL,NULL);
    return rc;
}


/*  hos_reverse(process id,procsize, tag, m, n, d, u[m], Z[n][d+1])            */
int hos_reverse( int id,
                 int size,
                 short tag,
                 int m,
                 int n,
                 int d,
                 double* u,
                 double** z
){
    int  rc=-3;
    if (id==0)
       rc = hos_reverse_mpi(id,size,tag,m,n,d,u,z);
    else
       rc = hos_reverse_mpi(id,size,tag,0,0,d,NULL,NULL);
    return rc;
}

/* fov_forward(process id, procsize, tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) */
int fov_forward( int id,
                 int size,
                 short tag,
                 int m,
                 int n,
                 int p,
                 const double* x,
                 double** a,
                 double* y,
                 double** b
){
    int rc=-3;
    if (id==0)
       rc = fov_forward_mpi(id,size,tag,m,n,p,x,a,y,b);
    else
       rc = fov_forward_mpi(id,size,tag,0,0,p,NULL,a,NULL,b);
    return rc;
}
/* fov_reverse(process id, procsize, tag, m, n, p, U[p][m], Z[p][n])  */
int fov_reverse( int id,
                 int size,
                 short tag,
                 int m,
                 int n,
                 int p,
                 double** u,
                 double** z
){
    int rc=-3;
    if (id==0)
       rc = fov_reverse_mpi(id,size,tag,m,n,p,u,z);
    else
       rc = fov_reverse_mpi(id,size,tag,0,0,p,NULL,NULL);
    return rc;
}

int hov_reverse(
      int id, int size, short tag, int m, int n, int degre,
      int nrows, double **lagrange, double ***results,
      short **nonzero){
int rc = -3;
    if (id==0)
       rc = hov_reverse_mpi(id,size,tag,m,n,degre,nrows, lagrange,results,nonzero);
    else
       rc = hov_reverse_mpi(id,size,tag,0,0,degre,nrows, NULL,NULL,NULL);
    return rc;
}

int hov_forward(
      int id, int size, short tag, int m, int n, int degre,
      int p,double *x, double ***a, double *v,
      double ***taylors){
int rc = -3;
    if (id==0)
       rc = hov_forward_mpi(id,size,tag,m,n,degre, p,x,a,v,taylors);
    else
       rc = hov_forward_mpi(id,size,tag,0,0,degre,p,
       NULL,NULL,NULL,NULL);
    return rc;
}

/* int_forward_tight(rank,size,tag, m, n, p, x[n], X[n][p], y[m], Y[m][p])            */
int int_forward_tight(
    int id,int size,short tag,
    int m,int n,int p, const double* x,
    unsigned long int** x_pp,double* y,unsigned long int** y_pp){

    int rc=-3;
    if (id==0)
       rc = int_forward_tight_mpi(id,size,tag,m,n,p,x,x_pp,y,y_pp);
    else
       rc = int_forward_tight_mpi(id,size,tag,0,0,p,NULL,NULL,NULL,NULL);
    return rc;
}

/* int_forward_safe(rank,size, tag, m, n, p, X[n][p], Y[m][p])                        */
int int_forward_safe(
    int id,int size,short tag,int m,int n,int p,unsigned long int **x_pp,unsigned long int **y_pp){

    int rc=-3;
    if (id==0)
       rc = int_forward_safe_mpi(id,size,tag,m,n,p,x_pp,y_pp);
    else
       rc = int_forward_safe_mpi(id,size,tag,0,0,p,NULL,NULL);
    return rc;
}

int int_reverse_tight(
    int id,int size,short tag, int m, int n, int p, unsigned long int **seed, unsigned long int **bit_pat){

    int rc=-3;
    if (id==0)
       rc = int_reverse_tight_mpi(id,size,tag,m,n,p,seed,bit_pat);
    else
       rc = int_reverse_tight_mpi(id,size,tag,0,0,p,NULL,NULL);
    return rc;
}
int int_reverse_safe(
    int id,int size,short tag, int m, int n, int p, unsigned long int **seed, unsigned long int **bit_pat){

    int rc=-3;
    if (id==0)
       rc = int_reverse_safe_mpi(id,size,tag,m,n,p,seed,bit_pat);
    else
       rc = int_reverse_safe_mpi(id,size,tag,0,0,p,NULL,NULL);
    return rc;
}


/* indopro_forward_tight(rank,size, tag, m, n, x[n], *crs[m])                         */
int indopro_forward_tight(
    int id, int size, short tag, int m, int n,const double *x, unsigned int **crs ){

    int rc=-3;
    if (id==0)
       rc = indopro_forward_tight_mpi(id,size,tag,m,n,x,crs);
    else
       rc = indopro_forward_tight_mpi(id,size,tag,0,0,x,NULL);
    return rc;
}

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])                                   */
int indopro_forward_safe(
   int id, int size, short tag, int m, int n,const double *x, unsigned int **crs ){

    int rc=-3;
    if (id==0)
       rc = indopro_forward_safe_mpi(id,size,tag,m,n,x,crs);
    else
       rc = indopro_forward_safe_mpi(id,size,tag,0,0,NULL,NULL);
    return rc;
}

/* indopro_forward_tight( tag, m, n, x[n], *crs[m])   */
int nonl_ind_forward_tight(
   int id, int size, short tag, int m, int n,const double *x, unsigned int **crs ){

    int rc=-3;
    if (id==0)
       rc = nonl_ind_forward_tight_mpi(id,size,tag,m,n,x,crs);
    else
       rc = nonl_ind_forward_tight_mpi(id,size,tag,0,0,NULL,NULL);
    return rc;
}

/* indopro_forward_safe( tag, m, n, x[n], *crs[m])   */
int nonl_ind_forward_safe(
   int id, int size, short tag, int m, int n,const double *x, unsigned int **crs ){

    int rc=-3;
    if (id==0)
       rc = indopro_forward_safe_mpi(id,size,tag,m,n,x,crs);
    else
       rc = indopro_forward_safe_mpi(id,size,tag,0,0,NULL,NULL);
    return rc;
}


