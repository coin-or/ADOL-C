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
#include <adolc/common.h>
#include <adolc/adalloc.h>

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
                  int dest, int tag, 
                  MPI_Comm comm ) {
	int i,h=count;
        double *trade;
        int ierr =0;

        put_op(receive_data);
        ADOLC_PUT_LOCINT(buf[0].loc());
        ADOLC_PUT_LOCINT(count);
        ADOLC_PUT_LOCINT(dest);
	ADOLC_PUT_LOCINT(tag);
        MPI_Status status;
        trade = (double*) myalloc1(h);
	ierr = MPI_Recv(trade,h, datatype, dest, tag, comm, &status);
        for(i=0; i< count;i++) 
	        buf[i].setValue(trade[i]);
        free(trade);
	return ierr;
}

/* That's all*/