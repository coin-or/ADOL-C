/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adolc_mpi.cpp
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Andrea Walther, Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <adolc/common.h>
#include <adolc/adolc_mpi.h>
#include "oplate.h"
#include "taping_p.h"
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/tapedoc/tapedoc.h>
#include <adolc/adalloc.h>
#include <adolc/interfaces_mpi.h>
#include <adolc/convolut.h>

int mpi_initialized = 0;
int all_root = 0;

int trace_on( int id,
              int size,
              short tag,
              int keepTaylors
){
    return trace_on(id+size*tag,keepTaylors);
}

BEGIN_C_DECLS

int ADOLC_MPI_Init( int* a,
                    char*** b
){
    mpi_initialized = 1;
    all_root = 0;
    return MPI_Init(a,b);
}
int ADOLC_MPI_Comm_size( ADOLC_MPI_Comm comm,
                         int* size
){
    int ierr = MPI_Comm_size(comm,size);
    return ierr;
}
int ADOLC_MPI_Comm_rank( ADOLC_MPI_Comm comm,
                         int* rank
){
    return MPI_Comm_rank(comm, rank);
}

int ADOLC_MPI_Get_processor_name( char* a,
                                  int* b
){
    return MPI_Get_processor_name(a,b);
}

int ADOLC_MPI_Barrier( ADOLC_MPI_Comm comm ){
    put_op(barrier_op);
    return MPI_Barrier(comm);
}

int ADOLC_MPI_Finalize( ){
    return MPI_Finalize();
}

END_C_DECLS

int ADOLC_MPI_Send( adouble *buf,
                    int count,
                    ADOLC_MPI_Datatype datatype,
                    int dest,
                    int tag,
                    ADOLC_MPI_Comm comm
){
    int i,h=count;
    int ierr =0;
    double *trade;

    trade = myalloc1(h);
    for (i=0; i< count;i++ )
        trade[i] = buf[i].getValue();

    ierr = MPI_Send(trade, h, datatype, dest, tag, comm);
    free(trade);

    put_op(send_data);
    ADOLC_PUT_LOCINT(count);
    if( buf[count-1].loc() - buf[0].loc() == count-1 ){
       ADOLC_PUT_LOCINT(1);
       ADOLC_PUT_LOCINT(buf[0].loc());
       ADOLC_PUT_LOCINT(1);
    }
    else {
    ADOLC_PUT_LOCINT(0);
      for (i=0; i< count;i++ )
       ADOLC_PUT_LOCINT(buf[i].loc());
    ADOLC_PUT_LOCINT(0);
    }
    ADOLC_PUT_LOCINT(count);
    ADOLC_PUT_LOCINT(dest);
    ADOLC_PUT_LOCINT(tag);

    return ierr;
}

int ADOLC_MPI_Recv( adouble *buf,
                   int count,
                   ADOLC_MPI_Datatype datatype,
                   int source, int tag,
                   ADOLC_MPI_Comm comm
) {
    int i,h=count;
    double *trade;
    int ierr =0;


    MPI_Status status;
    trade = (double*) myalloc1(h);
    ierr = MPI_Recv(trade,h, datatype, source, tag, comm, &status);

    if (buf==NULL)
       buf = new adouble[count];
    for (i=0; i< count;i++)
        buf[i].setValue(trade[i]);

    free(trade);

    put_op(receive_data);
    ADOLC_PUT_LOCINT(count);
    if( buf[count-1].loc() - buf[0].loc() == count-1 ){
       ADOLC_PUT_LOCINT(1);
       ADOLC_PUT_LOCINT(buf[0].loc());
       ADOLC_PUT_LOCINT(1);
    } else {
       ADOLC_PUT_LOCINT(0);
       for (i=0; i< count;i++ )
          ADOLC_PUT_LOCINT(buf[i].loc());
       ADOLC_PUT_LOCINT(0);
    }
    ADOLC_PUT_LOCINT(count);
    ADOLC_PUT_LOCINT(source);
    ADOLC_PUT_LOCINT(tag);

    return ierr;
}
int ADOLC_MPI_Bcast( adouble *buf,
                     int count,
                     ADOLC_MPI_Datatype datatype,
                     int root,
                     ADOLC_MPI_Comm comm )

{
    int i,id, ierr=0;
    double *trade;

    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    trade = (double*) myalloc1(count);

    if ( id == root)
       for(i= 0; i < count; i++)
          trade[i] = buf[i].getValue();

    ierr = MPI_Bcast(trade,count,datatype,root, comm);

    if ( id != root){
       if (buf==NULL)
          buf = new adouble[count];
       for(i=0; i< count;i++)
          buf[i].setValue(trade[i]);
    }

    free(trade);

    put_op(broadcast);
    ADOLC_PUT_LOCINT(count);
    if( buf[count-1].loc() - buf[0].loc() == count-1 ){
       ADOLC_PUT_LOCINT(1);
       ADOLC_PUT_LOCINT(buf[0].loc());
       ADOLC_PUT_LOCINT(1);
    } else {
       ADOLC_PUT_LOCINT(0);
       for (i=0; i< count;i++ )
          ADOLC_PUT_LOCINT(buf[i].loc());
       ADOLC_PUT_LOCINT(0);
    }
    ADOLC_PUT_LOCINT(count);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);

    return ierr;
}

int ADOLC_MPI_Reduce(
    adouble *send_buf, adouble *rec_buf, int count, ADOLC_MPI_Datatype type,
    ADOLC_MPI_Op op, int root, ADOLC_MPI_Comm comm)
{
    int i,j,id,size, ierr=0;

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    adouble *tmp_adoubles = NULL, tmp;
    if( id == root ){
         ensureContiguousLocations(size*count);
        tmp_adoubles = new adouble[size*count];
    }

    ierr = ADOLC_MPI_Gather(send_buf,tmp_adoubles,count,type,root,comm);
    if ( id == root){
       if( rec_buf == NULL){
           ensureContiguousLocations(count);
           rec_buf = new adouble[count];
       }
       switch (op) {
               case ADOLC_MPI_MAX: for(i=0; i < count; i++ ) {
                                       tmp = tmp_adoubles[i];
                                       for(j=1; j< size ; j++)
					   condassign(tmp, (adouble) (tmp <= tmp_adoubles[j*count+i]), tmp_adoubles[j*count+i] );
                                       rec_buf[i] = tmp;
                                   }
                                   break;
               case ADOLC_MPI_MIN: for(i=0; i < count; i++ ) {
                                      tmp = tmp_adoubles[i];
                                      for(j=1; j< size ; j++)
					  condassign(tmp, (adouble) (tmp >= tmp_adoubles[j*count+i]), tmp_adoubles[j*count+i] );
                                      rec_buf[i] = tmp;
                                   }
                                   break;
               case ADOLC_MPI_SUM: for(i=0; i < count; i++ ) {
                                      tmp =0.;
                                      for(j=0; j< size ; j++)
                                         tmp += tmp_adoubles[j*count+i];
                                       rec_buf[i] = tmp;
                                   }
                                   break;
               case ADOLC_MPI_PROD:for(i=0; i < count; i++ ) {
                                      tmp = 1.;
                                      for(j=0; j< size ; j++)
                                         tmp *= tmp_adoubles[j*count+i];
                                      rec_buf[i] = tmp;
                                    }
                                    break;
               default:             printf("Operation %d not yet implemented!\n",op);
                                    break;
       }
       delete[] tmp_adoubles;
    }

    return ierr;
}

int ADOLC_MPI_Gather(
    adouble *sendbuf, adouble *recvbuf, int count, ADOLC_MPI_Datatype type, int root, MPI_Comm comm)
{
    int i,id,size, ierr=0;
    double *trade_s, *trade_r;

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    trade_s = (double*) myalloc1(count);
    if (id == root)
      trade_r = (double*) myalloc1(count*size);
    else trade_r = NULL;

    for(i= 0; i < count; i++) {
       trade_s[i] = sendbuf[i].getValue();
    }
    ierr = MPI_Gather(trade_s,count,type,trade_r,count,type, root, comm);

    if ( id == root){
       if( recvbuf == NULL){
           ensureContiguousLocations(size*count);
           recvbuf = new adouble[count*size];
       }
       for(i=0; i< count*size;i++){
          recvbuf[i].setValue(trade_r[i]);
          }
    free(trade_r);
    }
    free( trade_s);

    put_op(gather);
    ADOLC_PUT_LOCINT(count);
    if( sendbuf[count-1].loc() - sendbuf[0].loc() == count-1 ){
       ADOLC_PUT_LOCINT(1);
       ADOLC_PUT_LOCINT(sendbuf[0].loc());
       ADOLC_PUT_LOCINT(1);
    } else {
       ADOLC_PUT_LOCINT(0);
       for(i= 0; i < count; i++)
        ADOLC_PUT_LOCINT(sendbuf[i].loc());
       ADOLC_PUT_LOCINT(0);
    }
    ADOLC_PUT_LOCINT(count);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);
    ADOLC_PUT_LOCINT(count*size);
    if( id==root){
       if( recvbuf[count*size-1].loc() - recvbuf[0].loc() == count*size-1 ){
         ADOLC_PUT_LOCINT(1);
         ADOLC_PUT_LOCINT(sendbuf[0].loc());
         ADOLC_PUT_LOCINT(1);
       } else {
         ADOLC_PUT_LOCINT(0);
         for(i=0; i < count*size;i++)
           ADOLC_PUT_LOCINT(recvbuf[i].loc());
         ADOLC_PUT_LOCINT(0);
       }
    }
    ADOLC_PUT_LOCINT(count*size);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);

    return ierr;
}

int ADOLC_MPI_Scatter(
    adouble *sendbuf, int sendcount, adouble *recvbuf,
    int recvcount, ADOLC_MPI_Datatype type, int root, MPI_Comm comm)
{
    int i,id,size, ierr=0;
    double *trade_s, *trade_r;

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    trade_r = (double*) myalloc1(recvcount);
    if (id == root)
      trade_s = (double*) myalloc1(sendcount*size);
    else trade_s = NULL;

    if ( id == root){
       for(i= 0; i < sendcount*size; i++)
          trade_s[i] = sendbuf[i].getValue();
    }

    ierr = MPI_Scatter(trade_s,sendcount,type,trade_r,recvcount,type, root, comm);

    if( recvbuf == NULL){
       ensureContiguousLocations(recvcount);
       recvbuf = new adouble[recvcount];
    }
    for(i=0; i< recvcount;i++)
         recvbuf[i].setValue(trade_r[i]);

    free(trade_s);
    free(trade_r);

    put_op(scatter);
    ADOLC_PUT_LOCINT(sendcount*size);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);
    if( id == root ) {
       if( sendbuf[sendcount*size-1].loc() - sendbuf[0].loc() == sendcount*size-1 ){
          ADOLC_PUT_LOCINT(1);
          ADOLC_PUT_LOCINT(sendbuf[0].loc());
          ADOLC_PUT_LOCINT(1);
       } else {
          ADOLC_PUT_LOCINT(0);
          for(i=0; i< sendcount*size ;i++)
             ADOLC_PUT_LOCINT(sendbuf[i].loc());
          ADOLC_PUT_LOCINT(0);
       }
    }
    ADOLC_PUT_LOCINT(sendcount*size);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);
    ADOLC_PUT_LOCINT(recvcount);
    if( recvbuf[recvcount-1].loc() - recvbuf[0].loc() == recvcount-1 ){
       ADOLC_PUT_LOCINT(1);
       ADOLC_PUT_LOCINT(recvbuf[0].loc());
       ADOLC_PUT_LOCINT(1);
    } else {
       ADOLC_PUT_LOCINT(0);
       for(i=0; i< recvcount;i++)
          ADOLC_PUT_LOCINT(recvbuf[i].loc());
       ADOLC_PUT_LOCINT(0);
    }
    ADOLC_PUT_LOCINT(recvcount);

    return ierr;
}


/*********************************************************************/
/* Algorithmic Differentation Programs                               */

int function(int id, int size,short tag,int m,int n,double* argument ,double* result){
     return function_mpi(id,size,tag,m,n,argument,result);
}

int gradient(int id,int size,short tag,int n,double *x,double *y)
{
     return gradient_mpi(id,size,tag,n,x,y);
}

int hessian(int id,int size,short tag,int n,double *x,double **x_pp)
{
     return hessian_mpi(id,size,tag,n,x,x_pp);
}

int jacobian(int id,int size,short tag,int m,int n,const double *x,double **x_pp)
{
return jacobian_mpi(id,size,tag,m,n,x,x_pp);
}

int vec_jac(int id,int size,short tag,int m,int n,int p,double *x,double *y, double *z)
{
return vec_jac_mpi(id,size,tag,m,n,p,x,y,z);
}

int jac_vec(int id,int size,short tag,int m,int n,double *x,double *y,double *z)
{
return jac_vec_mpi(id,size,tag,m,n,x,y,z);
}

int hess_vec(int id,int size,short tag,int n,double *x,double *y,double *z)
{
return hess_vec_mpi(id,size,tag,n,x,y,z);
}

int lagra_hess_vec(int id,int size,short tag,int n,int p,double *x,double *y,double *t,double *z)
{
return lagra_hess_vec_mpi(id,size,tag,n,p,x,y,t,z);
}

void tape_doc(int id,int size,short tag, int m,int n, double *x, double *y)
{
return tape_doc_mpi(id,size,tag,m,n,x,y);
}


int function_distrib(int id, int size,short tag,int m,int n,double* argument ,double* result){
     return function_mpi_distrib(id,size,tag,m,n,argument,result);
}

int gradient_distrib(int id,int size,short tag,int n,double *x,double *y)
{
     return gradient_mpi_distrib(id,size,tag,n,x,y);
}
void tape_doc_distrib(int id,int size,short tag, int m,int n, double *x, double *y)
{
return tape_doc_mpi_distrib(id,size,tag,m,n,x,y);
}

void tapestats(int id, int size, short tag, size_t *tape_stats)
{
     tapestats_mpi( id ,size, tag, tape_stats );
}

void printTapeStats(FILE *stream, short tag, int root)
{
     printTapeStats_mpi( stream, tag, root);
}

int removeTape(short tapeID, short type, int root) {
return removeTape_mpi( tapeID, type, root);
}

BEGIN_C_DECLS

/* C - functions                                   */
int function_mpi(int id, int size,short tag,int m,int n,double* argument ,double* result){
     int rc =-1;
     if( id == 0)
          rc = zos_forward_mpi(id,size,tag,m,n,0,argument,result);
     else
          rc = zos_forward_mpi(id,size,tag,0,0,0,NULL,NULL);
     return rc;
}

int gradient_mpi(int id,int size,short tag ,int n, double* x,double* result){

     int rc=-1;
     double one =1.0;
     if( id == 0){
          rc = zos_forward_mpi(id,size,tag,1,n,1,x,result);
          if(rc <0){
               printf("Failure by computing parallel gradient, process id %d!\n",id);
               return rc;
          }
          rc = fos_reverse_mpi(id,size,tag,1,n,&one,result);
     } else {
          rc = zos_forward_mpi(id,size,tag,0,0,1,NULL,NULL);
          if(rc <0){
               printf("Failure by computing parallel gradient, process id %d!\n",id);
               return rc;
          }
          rc = fos_reverse_mpi(id,size,tag,0,0,&one,NULL);
     }
     return rc;
}

int jacobian_mpi(int id, int size ,short tag ,int m,int n,const double* x,double** jacobian){
    int rc=-1;
    double *result = NULL , **I = NULL;

    if(id == 0){
        result = myalloc1(m);
        if (n/2 < m) {
            I = myallocI2(n);
            rc = fov_forward_mpi(id,size,tag,m,n,n,x,I,result,jacobian);
            myfreeI2(n, I);
        } else {
            I = myallocI2(m);
            rc = zos_forward_mpi(id,size,tag,m,n,1,x,result);
            if(rc <0){
               printf("Failure by computing parallel jacobian, process id %d!\n",id);
               return rc;
            }
            rc = fov_reverse_mpi(id,size,tag,m,n,m,I,jacobian);
            myfreeI2(m, I);
        }
        myfree1(result);
     } else {
        if (n/2 < m) {
            rc = fov_forward_mpi(id,size,tag,0,0,n,NULL,NULL,NULL,NULL);
        } else {
            rc = zos_forward_mpi(id,size,tag,0,0,1,NULL,NULL);
            if(rc <0){
               printf("Failure by computing parallel jacobian, process id %d!\n",id);
               return rc;
            }
            rc = fov_reverse_mpi(id,size,tag,0,0,m,NULL,NULL);
           }
     }
     return rc;
}

int hessian_mpi(int id,int size,short tag ,int n,double* x ,double** result){
     int rc =-3,i,j;
     if ( id == 0){
        double *v = myalloc1(n);
        double *w = myalloc1(n);
        for(i=0;i<n;i++) v[i] = 0;
        for(i=0;i<n;i++) {
           v[i] = 1;
           rc = hess_vec_mpi(id,size,tag, n, x, v, w);
           if(rc <0){
             printf("Failure by computing parallel hessian, process id %d!\n",id);
             free(v);
             free(w);
             return rc;
           }
           for(j=0;j<=i;j++)
            result[i][j] = w[j];
           v[i] = 0;
        }
        free(v);
        free(w);
     } else {
        for (i=0; i<n; i++){
            rc = fos_forward_mpi(id,size,tag, 0,0,2,NULL,NULL,NULL,NULL);
            if (rc <0){
               printf("Failure by computing parallel hessian, process id %d!\n",id);
               return rc;
            }
            rc = hos_reverse_mpi(id,size,tag,0,0,1, NULL,NULL);
        }
     }
     return rc;
}

/* vec_jac(rank,size,tag, m, n, repeat, x[n], u[m], v[n])                             */
int vec_jac_mpi( int id,int size,short tag,int m,int n,int repeat ,double *x,double *u,double *v){
     int rc = -3;
     double *y = NULL;
     if (id == 0){
        if(!repeat) {
            y = myalloc1(m);
            rc = zos_forward_mpi(id,size,tag,m,n,1, x, y);
            if (rc <0){
               printf("Failure by computing parallel vec_jac, process id %d!\n",id);
               return rc;
            }
        }
       MINDEC(rc, fos_reverse_mpi(id,size,tag,m,n,u,v));
       if (!repeat) myfree1(y);
     } else{
        if(!repeat) {
           rc = zos_forward_mpi(id,size,tag,0,0,1,NULL,NULL);
        if(rc < 0) return rc;
        }
        rc = fos_reverse_mpi(id,size,tag,0,0,NULL,NULL);
     }
     return rc;
}

/* jac_vec(rank,size,tag, m, n, x[n], v[n], u[m]);                                    */
int jac_vec_mpi(int id,int size,short tag,int m,int n,double *x,double *v, double *u){
     int rc = -3;
     double *y = NULL;
     if (id == 0){
         y = myalloc1(m);
         rc = fos_forward_mpi(id,size,tag, m, n, 0, x, v, y, u);
         myfree1(y);
     } else{
         rc = fos_forward_mpi(id,size,tag, 0, 0, 0, NULL, NULL, NULL,NULL);
     }
     return rc;
}

/* hess_vec(rank,size,tag, n, x[n], v[n], w[n])                                       */
int hess_vec_mpi(int id,int size,short tag,int n,double *x,double *v,double *w){
     double one = 1.0;
     return lagra_hess_vec_mpi(id,size,tag,1,n,x,v,&one,w);
}

/* lagra_hess_vec(tag, m, n, x[n], v[n], u[m], w[n])                        */
int lagra_hess_vec_mpi(int id, int size, short tag,
                   int m,
                   int n,
                   double *argument,
                   double *tangent,
                   double *lagrange,
                   double *result) {
    int rc=-1;
    int i;
    int degree = 1;
    int keep = degree+1;
    double **X, *y, *y_tangent;

     if (id == 0 ){
        X = myalloc2(n,2);
        y = myalloc1(m);
        y_tangent = myalloc1(m);
        rc = fos_forward_mpi(id,size,tag, m, n, keep, argument, tangent, y, y_tangent);
        if (rc <0){
           printf("Failure by computing parallel lagra_hess_vec, process id %d!\n",id);
           return rc;
        }
        MINDEC(rc, hos_reverse_mpi(id,size,tag, m, n, degree, lagrange, X));
        for(i = 0; i < n; ++i)
            result[i] = X[i][1];
        myfree1(y_tangent);
        myfree1(y);
        myfree2(X);
     } else {
        rc = fos_forward_mpi(id,size,tag, 0, 0, keep, NULL, lagrange , NULL, NULL);
        if (rc <0){
           printf("Failure by computing parallel lagra_hess_vec, process id %d!\n",id);
           return rc;
        }
        rc = hos_reverse_mpi(id,size,tag, 0, 0, degree, lagrange, NULL );
    }
     return rc;
}

/* routines for user defined dependends and independends    */
int function_mpi_distrib(int id, int size,short tag,int m,int n,double* argument ,double* result){
       return zos_forward_mpi(id,size,tag,m,n,0,argument,result);
}

int gradient_mpi_distrib(int id,int size,short tag ,int n, double* x,double* result){
     int rc=-1;
     double one =1.0;
     rc = zos_forward_mpi(id,size,tag,1,n,1,x,result);
     if(rc <0){
         printf("Failure by computing parallel gradient, process id %d!\n",id);
           return rc;
     }
     rc = fos_reverse_mpi(id,size,tag,1,n,&one,result);
     return rc;
}

void tape_doc_mpi( int id,int size,short tag, int m,int n, double* x, double* y){
     if(id==0)
        tape_doc(id+size*tag,m,n,x,y);
     else
        tape_doc(id+size*tag,0,0,x,y);
}

void tape_doc_mpi_distrib( int id,int size,short tag, int m,int n, double* x, double* y){
        tape_doc(id+size*tag,m,n,x,y);
}

void tapestats_mpi(int id, int size, short tag, size_t *tape_stats)
{
     tapestats( id +size*tag, tape_stats );
}

void printTapeStats_mpi(FILE *stream, short tag, int root)
{
  int id, size, ierr=0, curr_tag, i;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if( id == root ){
       fprintf(stream, "\n*** TAPE STATS (tape %d) **********\n", (int)tag);
       fprintf(stream, "\n*** Used Processes %d *************\n", size);
       for(i= 0; i < size; i++ ){
            size_t stats[STAT_SIZE];
            curr_tag = i + tag*size;
            tapestats( curr_tag , (size_t *)&stats);
            fprintf(stream, "\n*** Process ID %d               ***\n", i);
            fprintf(stream, "Number of independents: %10zd\n", stats[NUM_INDEPENDENTS]);
            fprintf(stream, "Number of dependents:   %10zd\n", stats[NUM_DEPENDENTS]);
            fprintf(stream, "\n");
            fprintf(stream, "Max # of live adoubles: %10zd\n", stats[NUM_MAX_LIVES]);
            fprintf(stream, "Taylor stack size:      %10zd\n", stats[TAY_STACK_SIZE]);
            fprintf(stream, "\n");
            fprintf(stream, "Number of operations:   %10zd\n", stats[NUM_OPERATIONS]);
            fprintf(stream, "Number of locations:    %10zd\n", stats[NUM_LOCATIONS]);
            fprintf(stream, "Number of values:       %10zd\n", stats[NUM_VALUES]);
            fprintf(stream, "\n");
            fprintf(stream, "Operation file written: %10zd\n", stats[OP_FILE_ACCESS]);
            fprintf(stream, "Location file written:  %10zd\n", stats[LOC_FILE_ACCESS]);
            fprintf(stream, "Value file written:     %10zd\n", stats[VAL_FILE_ACCESS]);
            fprintf(stream, "\n");
            fprintf(stream, "Operation buffer size:  %10zd\n", stats[OP_BUFFER_SIZE]);
            fprintf(stream, "Location buffer size:   %10zd\n", stats[LOC_BUFFER_SIZE]);
            fprintf(stream, "Value buffer size:      %10zd\n", stats[VAL_BUFFER_SIZE]);
            fprintf(stream, "Taylor buffer size:     %10zd\n", stats[TAY_BUFFER_SIZE]);
            fprintf(stream, "\n");
            fprintf(stream, "Operation type size:    %10zd\n", (size_t)sizeof(unsigned char));
            fprintf(stream, "Location type size:     %10zd\n", (size_t)sizeof(locint));
            fprintf(stream, "Value type size:        %10zd\n", (size_t)sizeof(double));
            fprintf(stream, "Taylor type size:       %10zd\n", (size_t)sizeof(revreal));
            fprintf(stream, "***                            ***\n\n");
       }
  }
}

int removeTape_mpi(short tapeID, short type, int root) {
  int id, size, ierr=0, i;
  MPI_Comm_rank(MPI_COMM_WORLD, &id);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if( id == root )
     for(i = 0 ; i < size; i++ )
        ierr = removeTape(i + size*tapeID , type);
     return ierr;
}

END_C_DECLS

/* That's all*/
