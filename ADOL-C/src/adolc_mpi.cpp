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
#include <adolc/oplate.h>
#include "taping_p.h"
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/tapedoc/tapedoc.h>
#include <adolc/adalloc.h>
#include <adolc/interfaces.h>
#include <adolc/convolut.h>

#define ADOLC_MPI_Datatype MPI_Datatype
#define MPI_ADOUBLE MPI_DOUBLE
#define ADOLC_MPI_COMM_WORLD MPI_COMM_WORLD
#define ADOLC_MPI_Comm MPI_Comm

int mpi_initialized = 0;
int process_count = 1;

int trace_on( int id,
              int size,
              short tag
){
    int this_tag = size*tag + id;
    return trace_on( this_tag );
}
int ADOLC_MPI_Init( int* a,
                    char*** b
){
    mpi_initialized = 1;
    return MPI_Init(a,b);
}
int ADOLC_MPI_Comm_size( ADOLC_MPI_Comm comm,
                         int* size
){
    int ierr = MPI_Comm_size(comm,size);
    process_count = size[0];
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

    trade = (double*) myalloc1(h);
    for (i=0; i< count;i++ )
        trade[i] = buf[i].getValue();

    ierr = MPI_Send(trade, h, datatype, dest, tag, comm);
    free(trade);

    put_op(send_data);
    ADOLC_PUT_LOCINT(buf[0].loc());
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
    ADOLC_PUT_LOCINT(buf[0].loc());
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
    int i,id,size, ierr=0;
    double *trade;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
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
    ADOLC_PUT_LOCINT(buf[0].loc()); // send
    ADOLC_PUT_LOCINT(buf[0].loc()); // recv
    ADOLC_PUT_LOCINT(count);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);

    return ierr;
}

int ADOLC_MPI_Reduce(
    adouble *send_buf, adouble *rec_buf, int count, ADOLC_MPI_Datatype datatype,
    ADOLC_MPI_Op op, int root, ADOLC_MPI_Comm comm)
{
    int i,id,size, ierr=0;
    double *trade_s, *trade_r;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    trade_s = (double*) myalloc1(count);
    if (id == root)
      trade_r = (double*) myalloc1(count);
    else trade_r = NULL;

    for(i= 0; i < count; i++) {
       trade_s[i] = send_buf[i].getValue();
     }
    ierr = MPI_Reduce(trade_s,trade_r ,count,datatype,op,root, comm);

    if ( id == root){
       if( rec_buf == NULL)
           rec_buf = new adouble[count];
       for(i=0; i< count;i++){
          rec_buf[i].setValue(trade_r[i]);
          }
    }
    free(trade_s);
    free(trade_r);

    put_op(reduce);
    ADOLC_PUT_LOCINT(send_buf[0].loc());
    ADOLC_PUT_LOCINT(rec_buf[0].loc());
    ADOLC_PUT_LOCINT(count);
    ADOLC_PUT_LOCINT(root);
    ADOLC_PUT_LOCINT(id);
    ADOLC_PUT_LOCINT(op);

    return ierr;
}

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
    int this_tag = size*tag + id, rc=-3;
    if (id==0)
       rc = zos_forward(this_tag,m,n,keep,x,y);
    else
       rc = zos_forward(this_tag,0,0,keep,NULL,NULL);
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
    int this_tag = size*tag + id;
    int rc=-3;
    if (id==0)
       rc = fos_forward(this_tag,m,n,keep,x,a,y,b);
    else
       rc = fos_forward(this_tag,0,0,keep,NULL,NULL,NULL,NULL);
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
    int this_tag = size*tag + id, rc=-3;
    if (id==0)
       rc = fos_reverse(this_tag,m,n,u,z);
    else
       rc = fos_reverse(this_tag,0,0,NULL,NULL);
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
    int this_tag = size*tag + id, rc=-3;
    if (id==0)
       rc = hos_forward(this_tag,depen,indep,d,keep,basepoints,argument,valuepoints,taylors);
    else
       rc = hos_forward(this_tag,0,0,d,keep,NULL,NULL,NULL,NULL);
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
    int this_tag = size*tag + id, rc=-3;
    if (id==0)
       rc = hos_reverse(this_tag,m,n,d,u,z);
    else
       rc = hos_reverse(this_tag,0,0,d,NULL,NULL);
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
    int this_tag = size*tag + id;
    int rc=-3;
    if (id==0)
       rc = fov_forward(this_tag,m,n,p,x,a,y,b);
    else
       rc = fov_forward(this_tag,0,0,p,NULL,a,NULL,b);
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
    int this_tag = size*tag + id;
    int rc=-3;
    if (id==0)
       rc = fov_reverse(this_tag,m,n,p,u,z);
    else
       rc = fov_reverse(this_tag,0,0,p,NULL,NULL);
    return rc;
}

/****************** Now algorithmic functions ****************/

int function(int id, int size,short tag,int m,int n,double* argument ,double* result){
	int rc =-1;
	int this_tag = tag*size + id;
	if( id == 0)
		rc = function(this_tag,m,n,argument,result);
	else
		rc = function(this_tag,0,0,NULL,NULL);
	return rc;
}

int gradient(int id,int size,short tag ,int n, double* x,double* result){
	int rc=-1;
	int this_tag = tag*size + id;
	double one =1.0;
	if( id == 0)
		rc = gradient(this_tag , n , x , result);
	else {
		rc = zos_forward(this_tag,0,0,1,NULL,NULL);
		if(rc <0){
			printf("Failure by computing parallel hessian, process id %d!\n",id);
			return rc;
		}
		rc = fos_reverse(this_tag,0,0,&one,result);
	}
	return rc;
}

int jacobian(int id, int size ,short tag ,int m,int n,const double* a,double** result){
	int rc=-1;
	int this_tag = size*tag + id;

	if(id==0){
		rc = jacobian(this_tag,m,n,a,result);
	} else {
		if (n/2 < m) {
			rc = fov_forward(this_tag,0,0,n,NULL,NULL,NULL,NULL);
		} else {
			rc = zos_forward(this_tag,0,0,1,a,NULL);
			if(rc <0){
				printf("Failure by computing parallel jacobian, process id %d!\n",id);
				return rc;
			}
			rc = fov_reverse(this_tag,0,0,m,NULL,result);
		}
	}
	return rc;
}

int hessian(int id,int size,short tag ,int n,double* x ,double** result){
	int rc =-3,i;
	int this_tag = tag*size + id;
	if ( id == 0){
         rc = hessian(this_tag,n,x,result);
	}
	else {
        for (i=0; i<n; i++){
            rc = fos_forward(this_tag, 0,0,2,NULL,NULL,NULL,NULL);
            if (rc <0){
               printf("Failure by computing parallel hessian, process id %d!\n",id);
               return rc;
            }
            rc = hos_reverse(this_tag,0,0,1, NULL,NULL);
        }
     }
     return rc;
}

void tape_doc( int id,int size,short tag, int m,int n, double* x, double* y){
	int this_tag = tag*size +id;
	if(id==0) tape_doc(this_tag,m,n,x,y);
	else tape_doc(this_tag,0,0,x,y);
}

/* That's all*/
