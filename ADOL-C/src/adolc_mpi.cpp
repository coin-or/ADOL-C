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

#include <adolc/adolc_mpi.h>
#include <adolc/oplate.h>
#include "taping_p.h"
#include <mpi/mpi.h>
#include <adolc/adouble.h>
#include <adolc/drivers/drivers.h>
#include <adolc/tapedoc/tapedoc.h>
#include <adolc/adalloc.h>
#include <adolc/common.h>
#include <adolc/interfaces.h>
#include <adolc/convolut.h>

int trace_on(int id, int size, short tag){
	int result = size*tag + id;
	return trace_on( result );
}

int ADOLC_MPI_Init(int* a, char*** b){
	return MPI_Init(a,b);
}
int ADOLC_MPI_Comm_size(MPI_Comm comm, int* size) {
	return MPI_Comm_size(comm,size);
}
int ADOLC_MPI_Comm_rank(MPI_Comm comm , int* rank) {
	return MPI_Comm_rank(comm, rank);
}
	
int ADOLC_MPI_Get_processor_name(char* a, int* b) {
	return MPI_Get_processor_name(a,b);
}
	
int ADOLC_MPI_Barrier(MPI_Comm comm) {
	return MPI_Barrier(comm);
}

int ADOLC_MPI_Finalize() {
	return MPI_Finalize();
}

int ADOLC_MPI_Send(adouble *buf, 
                  int count, 
                  MPI_Datatype datatype, 
                  int dest, int tag, 
                  MPI_Comm comm ){
	int i,h=count;
 	int ierr =0;
        double *trade;
        put_op(send_data);
        ADOLC_PUT_LOCINT(buf[0].loc());
        ADOLC_PUT_LOCINT(count);
        ADOLC_PUT_LOCINT(dest);
        ADOLC_PUT_LOCINT(tag);
        
        trade = (double*) myalloc1(h);
	for(i=0; i< count;i++)
		trade[i] = buf[i].getValue();

        ierr = MPI_Send(trade, h, datatype, dest, tag, comm);
        free(trade);
        return ierr;
}

int ADOLC_MPI_Recv(adouble *buf, 
                  int count, 
                  MPI_Datatype datatype, 
                  int source, int tag, 
                  MPI_Comm comm ) {
	int i,h=count;
        double *trade;
        int ierr =0;
	
        put_op(receive_data);
        ADOLC_PUT_LOCINT(buf[0].loc());
        ADOLC_PUT_LOCINT(count);
        ADOLC_PUT_LOCINT(source);
	ADOLC_PUT_LOCINT(tag);
        MPI_Status status;
        trade = (double*) myalloc1(h);
        ierr = MPI_Recv(trade,h, datatype, source, tag, comm, &status);
        if(buf==NULL) buf = new adouble[count];
        for(i=0; i< count;i++) 
	        buf[i].setValue(trade[i]);
        free(trade);
	return ierr;
}

int gradient(int id,int size,short tag ,int n, double* x,double* result){
	int rc=-1;
	int this_tag = tag*size + id;
	double one =1.0;
	if( id == 0)
		rc = gradient(this_tag , n , x , result);
	else {
		rc = zos_forward(this_tag,0,0,1,x,result);
		if(rc<0) return rc;
		rc = fos_reverse(this_tag,0,0,&one,result);
	}
	return rc;
}

int hessian(int id,int size,short tag ,int n,double* x ,double** result){
	int rc =-3,i;
	int this_tag = tag*size + id;
	double one = 1.0;
	if( id == 0)
		rc = hessian(this_tag,n,x,result);
	else {
		for(i=0;i<n;++i){
			rc = fos_forward(this_tag, 0,0,2,NULL,NULL,NULL,NULL);
			if(rc <0){
				printf("Failure by computing parallel hessian, process id %d!\n",id);
				return rc;
			}
			rc = hos_reverse(this_tag,0,0,1, NULL,NULL);
		}
	}
	return rc;
}

/* That's all*/