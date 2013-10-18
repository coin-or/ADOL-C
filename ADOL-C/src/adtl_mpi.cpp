/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     adtl_mpi.cpp
 Revision: $Id$
 Contents: C allocation of arrays of doubles in several dimensions

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include <iostream>

#include <adolc/adtl_mpi.h>

namespace adtl{

int ADTL_MPI_Init( int* a,
                    char*** b
){
    return MPI_Init(a,b);
}
int ADTL_MPI_Comm_size( ADTL_MPI_Comm comm,
                         int* size
){
    int ierr = MPI_Comm_size(comm,size);
    return ierr;
}
int ADTL_MPI_Comm_rank( ADTL_MPI_Comm comm,
                         int* rank
){
    return MPI_Comm_rank(comm, rank);
}

int ADTL_MPI_Get_processor_name( char* a,
                                  int* b
){
    return MPI_Get_processor_name(a,b);
}

int ADTL_MPI_Barrier( ADTL_MPI_Comm comm ){
    return MPI_Barrier(comm);
}

int ADTL_MPI_Finalize( ){
    return MPI_Finalize();
}

MPI_Op adolc_to_mpi_op(ADTL_MPI_Op op) {
    switch (op) {
	case ADTL_MPI_MAX: return MPI_MAX;
	case ADTL_MPI_MIN: return MPI_MIN;
	case ADTL_MPI_SUM: return MPI_SUM;
	case ADTL_MPI_PROD: return MPI_PROD;
	case ADTL_MPI_LAND: return MPI_LAND;
	case ADTL_MPI_BAND: return MPI_BAND;
	case ADTL_MPI_LOR: return MPI_LOR;
	case ADTL_MPI_BOR: return MPI_BOR;
	case ADTL_MPI_LXOR: return MPI_LXOR;
	case ADTL_MPI_BXOR: return MPI_BXOR;
	case ADTL_MPI_MINLOC: return MPI_MINLOC;
	case ADTL_MPI_MAXLOC: return MPI_MAXLOC;
    }
}


void ADTL_MPI_set_trade(adouble *buf, int count, size_t nd, double *trade){
    int l=0;
     for (int i=0; i< count; i++ ){
       trade[l] = buf[i].getValue();
       l++;
       if (do_adval()) {
         for (unsigned int j=0; j< nd ;j++,l++)
            trade[l] = buf[i].getADValue(j);
       }
    }
}

void ADTL_MPI_get_trade(adouble *buf, int count, size_t nd, double *trade){
    int l=0;
    for (int i=0; i< count;i++){
       buf[i].setValue(trade[l]);
       l++;
       if (do_adval()) {
         for (unsigned int j=0; j< nd ;j++,l++)
            buf[i].setADValue(j,trade[l]);
       }
    }
}
void ADTL_MPI_set_trade_uint(adouble *buf, int count, size_t nd, unsigned int *trade) {
    if (trade == NULL)
         trade = new unsigned int[count * ( nd+1 )];
    unsigned int l =0, k;
    for (int i=0; i< count; i++ ){
          trade[l] = buf[i].get_pattern_size();
       if( trade[l] > 0 ){
          const list<unsigned int>& tmp = buf[i].get_pattern();
          list<unsigned int>::const_iterator it;
          k=1;
          for( it = tmp.begin() ; it != tmp.end() ; it++,k++)
             trade[l+k] = *it;
       }
       l += nd +1;
    }
}

void ADTL_MPI_get_trade_uint(adouble *buf, int count, size_t nd, unsigned int *trade){
     unsigned int l=0;
     for (int i=0; i< count; i++ ){
        for( unsigned int j = 1; j < trade[l]+1 ; j++)
           buf[i].pattern.push_back( trade[l+j] );
        l += nd+1;
        }
}

int ADTL_MPI_Send( adouble *buf,
                    int count,
                    ADTL_MPI_Datatype datatype,
                    int dest,
                    int tag,
                    ADTL_MPI_Comm comm
){
    size_t mpi_numdir = adouble::numDir;
    unsigned int i,j,h=count*(1+mpi_numdir);
    int ierr =0, l;

    double *trade = new double[h];
    unsigned int *trade_uint=NULL;

    ADTL_MPI_set_trade(buf,count,mpi_numdir, trade);

    ierr = MPI_Send(trade, h, datatype, dest, tag, comm);
    delete[] trade;

    if (do_indo()) {
       size_t numd =0;
       for (i=0; i< count; i++ )
          if ( buf[i].get_pattern_size() > numd)
             numd = buf[i].get_pattern_size();
       ierr = MPI_Send(&numd, 1, MPI_UNSIGNED, dest, tag, comm);

       if( numd > 0){
         trade_uint = NULL;
         h = count * (numd+1);
         trade_uint = new unsigned int[h];
         ADTL_MPI_set_trade_uint(buf,count,numd,trade_uint);
         ierr = MPI_Send(trade_uint, h , MPI_UNSIGNED, dest, tag, comm);
         delete[] trade_uint;
       }
    }
    return ierr;
}

int ADTL_MPI_Recv( adouble *buf,
                   int count,
                   ADTL_MPI_Datatype datatype,
                   int source, int tag,
                   ADTL_MPI_Comm comm
) {
    if (buf==NULL)
       buf = new adouble[count];
    size_t mpi_numdir = adouble::numDir;
    unsigned int i,j,h=count*(1+mpi_numdir);
    int ierr =0, l;

    double *trade = new double[h];
    unsigned int *trade_uint = NULL;

    MPI_Status status;
    ierr = MPI_Recv(trade,h, datatype, source, tag, comm, &status);
    ADTL_MPI_get_trade(buf, count, mpi_numdir, trade);
    delete[] trade;

    if (do_indo()) {
       size_t numd;
       ierr = MPI_Recv(&numd ,1, MPI_UNSIGNED , source, tag, comm, &status);
       if( numd > 0){
         h = count*(numd+1);
         trade_uint = new unsigned int[h];
         ierr = MPI_Recv(trade_uint,h, MPI_UNSIGNED , source, tag, comm, &status);
         ADTL_MPI_get_trade_uint(buf, count, numd, trade_uint);
         delete[] trade_uint;
       }
    }
    return ierr;
}

int ADTL_MPI_Bcast( adouble *buf,
                     int count,
                     ADTL_MPI_Datatype datatype,
                     int root,
                     ADTL_MPI_Comm comm )

{
    if (buf==NULL)
      buf = new adouble[count];
    size_t mpi_numdir = adouble::numDir;
    unsigned int i,j,h=count*(1+mpi_numdir);
    int ierr =0, l;

    double *trade = new double[h];
    unsigned int *trade_uint=NULL;
    l=0;
    int id;
    MPI_Comm_rank(comm, &id);

    if (id == root)
        ADTL_MPI_set_trade(buf,count,mpi_numdir, trade);

    ierr = MPI_Bcast(trade,h,datatype,root, comm);

    if ( id != root)
       ADTL_MPI_get_trade(buf,count,mpi_numdir, trade);

    delete[] trade;
    if (do_indo()) {
       size_t numd;
       if (root==id){
         numd =0;
         for (i=0; i< count; i++ )
          if ( buf[i].get_pattern_size() > numd)
             numd = buf[i].get_pattern_size();
       }
       ierr = MPI_Bcast(&numd,1,MPI_UNSIGNED, root, comm);
       if( numd > 0){
         h = count*(numd+1);
         trade_uint = new unsigned int[h];
         if ( id == root)
            ADTL_MPI_set_trade_uint(buf,count, numd, trade_uint);
         ierr = MPI_Bcast(trade_uint,h,MPI_UNSIGNED, root, comm);
         if(id != root)
            ADTL_MPI_get_trade_uint(buf,count,numd, trade_uint);
         delete[] trade_uint;
       }
    }
    return ierr;
}

int ADTL_MPI_Reduce(
    adouble *send_buf, adouble *rec_buf, int count, ADTL_MPI_Datatype type,
    ADTL_MPI_Op op, int root, ADTL_MPI_Comm comm)
{
    int i,j,id,size, ierr=0;
    adouble tmp, *tmp_adoubles = NULL;
    double *trade_s, *trade_r;

    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &size);
    ierr = ADTL_MPI_Gather(send_buf,tmp_adoubles,count,type,root,comm);

    if ( id == root){
       if( rec_buf == NULL)
           rec_buf = new adouble[count];
       switch (op) {
               case ADTL_MPI_MAX: for(i=0; i < count; i++ ) {
                                       tmp = tmp_adoubles[i];
                                       for(j=1; j< size ; j++)
                                          if ( tmp <= tmp_adoubles[j*count+i] )
                                             tmp = tmp_adoubles[j*count+i];
                                       rec_buf[i] = tmp;
                                   }
                                   break;
               case ADTL_MPI_MIN: for(i=0; i < count; i++ ) {
                                      tmp = tmp_adoubles[i];
                                      for(j=1; j< size ; j++)
                                         if ( tmp >= tmp_adoubles[j*count+i] )
                                            tmp = tmp_adoubles[j*count+i];
                                      rec_buf[i] = tmp;
                                   }
                                   break;
               case ADTL_MPI_SUM: for(i=0; i < count; i++ ) {
                                      tmp =0.;
                                      for(j=0; j< size ; j++)
                                         tmp += tmp_adoubles[j*count+i];
                                       rec_buf[i] = tmp;
                                   }
                                   break;
               case ADTL_MPI_PROD:for(i=0; i < count; i++ ) {
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

int ADTL_MPI_Gather( adouble *sendbuf, adouble *recvbuf, int count,
                      ADTL_MPI_Datatype type, int root, MPI_Comm comm
){
    size_t mpi_numdir = adouble::numDir;
    unsigned int i,j,h=count*(1+mpi_numdir);
    int ierr =0, l;

    int id,size;
    double *trade_s, *trade_r;
    unsigned int *trade_suint=NULL, *trade_ruint;

    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &size);

    trade_s = new double[h];
    if (id == root){
      trade_r = new double[h*size];
      if (recvbuf == NULL)
          recvbuf = new adouble[count*size];
    }
    else trade_r = NULL;
    ADTL_MPI_set_trade(sendbuf,count,mpi_numdir, trade_s);
    ierr = MPI_Gather(trade_s,h,type,trade_r,h,type, root, comm);
    if ( id == root){
       ADTL_MPI_get_trade(recvbuf,count*size,mpi_numdir, trade_r);
       delete[] trade_r;
    }
    delete[] trade_s ;

    if (do_indo()) {
         size_t numd =0;
         unsigned int tmp_nd =0;
         for (i=0; i< count; i++ )
          if ( sendbuf[i].get_pattern_size() > numd)
             numd = sendbuf[i].get_pattern_size();
         ierr = MPI_Reduce(&numd,&tmp_nd,1,MPI_UNSIGNED,MPI_MAX, root, comm);
         ierr = MPI_Bcast(&tmp_nd,1,MPI_UNSIGNED, root, comm);

         if( tmp_nd > 0){
           h = count*(numd+1);
           if( id == root )
              trade_ruint = new unsigned int[h*size];
           else trade_ruint = NULL;
           trade_suint = new unsigned int[h];

           ADTL_MPI_set_trade_uint(sendbuf,count ,numd, trade_suint);

           ierr = MPI_Gather(trade_suint, h,MPI_UNSIGNED,trade_ruint, h,MPI_UNSIGNED, root, comm);
           if (id == root){
             ADTL_MPI_get_trade_uint(recvbuf,count*size,numd, trade_ruint);
             delete[] trade_ruint;
           }
           delete[] trade_suint;
       }
    }
    return ierr;
}

int ADTL_MPI_Scatter(
    adouble *sendbuf, int sendcount, adouble *recvbuf,
    int recvcount, ADTL_MPI_Datatype type, int root, MPI_Comm comm)
{
    size_t mpi_numdir = adouble::numDir;
    unsigned int i,j,h=sendcount*(1+mpi_numdir);
    int ierr =0, l;

    int id,size;
    double *trade_s, *trade_r;
    unsigned int *trade_suint = NULL , *trade_ruint;

    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &size);

    if (id == root)
      trade_s = new double[h*size];
    else trade_s = NULL;
    trade_r = new double[recvcount*(1+mpi_numdir)];

    if ( id == root)
       ADTL_MPI_set_trade(sendbuf,sendcount*size,mpi_numdir, trade_s);

    ierr = MPI_Scatter(trade_s,sendcount,type,trade_r,recvcount,type, root, comm);

    ADTL_MPI_get_trade(recvbuf,recvcount,mpi_numdir, trade_r);

    delete[] trade_s,trade_r;

    if (do_indo()) {
         size_t numd =0;
         size_t tmp_nd =0;
         if ( id == root)
           for (i=0; i< sendcount; i++ )
             if ( sendbuf[i].get_pattern_size() > numd)
                numd = sendbuf[i].get_pattern_size();

        ierr = MPI_Bcast(&numd,1,MPI_UNSIGNED, root,comm);

       if( numd > 0){
         trade_ruint = new unsigned int[(numd+1)*recvcount];
         if( id == root ){
            trade_suint = new unsigned int[sendcount*(numd+1)*size];
            ADTL_MPI_set_trade_uint(sendbuf,sendcount*size,numd, trade_suint);
         }

         ierr = MPI_Scatter(trade_suint,sendcount*(numd+1),MPI_UNSIGNED,trade_ruint,recvcount*(numd+1),MPI_UNSIGNED, root, comm);
         if (id == root)
            delete[] trade_suint;

         ADTL_MPI_get_trade_uint(recvbuf,recvcount,numd, trade_ruint);
         delete[] trade_ruint;
       }
    }
    return ierr;
}

int ADTL_MPI_Allgather(
    adouble *sendbuf, int sendcount,ADTL_MPI_Datatype stype, adouble *recvbuf, int recvcount,
    ADTL_MPI_Datatype rtype, ADTL_MPI_Comm comm)
{
    size_t mpi_numdir = adouble::numDir;
    unsigned int i,j,h=sendcount*(1+mpi_numdir);
    int ierr =0, l;

    int id,size;
    double *trade_s, *trade_r;
    unsigned int *trade_suint=NULL , *trade_ruint;

    MPI_Comm_rank(comm, &id);
    MPI_Comm_size(comm, &size);

    trade_s = new double[h];
    trade_r = new double[h*size];
    if (recvbuf == NULL)
       recvbuf = new adouble[recvcount*size];

    ADTL_MPI_set_trade(sendbuf,sendcount,mpi_numdir, trade_s);

    ierr = MPI_Allgather(trade_s,h,stype, trade_r, h, rtype, comm);

    ADTL_MPI_get_trade(recvbuf,recvcount*size,mpi_numdir, trade_r);

    delete[] trade_s,trade_r;

    if (do_indo()) {
         size_t numd =0;
         size_t tmp_nd =0;
         for (i=0; i< sendcount; i++ )
             if ( sendbuf[i].get_pattern_size() > numd)
                numd = sendbuf[i].get_pattern_size();
       ierr = MPI_Allreduce(&tmp_nd,&numd,1,MPI_UNSIGNED,MPI_MAX, comm);

       if( numd > 0){
         trade_ruint = new unsigned int[(numd+1)*recvcount*size];
         trade_suint = new unsigned int[(numd+1)*sendcount];

         ADTL_MPI_set_trade_uint(sendbuf,sendcount,numd, trade_suint);
         ierr = MPI_Allgather(trade_suint,sendcount*(numd+1),MPI_UNSIGNED,trade_ruint,recvcount*(numd+1),MPI_UNSIGNED, comm);
         ADTL_MPI_get_trade_uint(recvbuf,recvcount*size, numd, trade_ruint);

         delete[] trade_ruint;
         delete[] trade_suint;
       }
    }
    return ierr;
}

int ADTL_MPI_Allreduce(
    adouble *send_buf, adouble *rec_buf, int count, ADTL_MPI_Datatype type,
    ADTL_MPI_Op op, ADTL_MPI_Comm comm)
{
    int i,j,size, ierr=0;
    MPI_Comm_size(comm, &size);
    adouble tmp, *tmp_adoubles = new adouble[count*size];
    ierr = ADTL_MPI_Allgather(send_buf,count,type,tmp_adoubles,count,type,comm);

    if( rec_buf == NULL)
       rec_buf = new adouble[count];
    switch (op) {
               case ADTL_MPI_MAX: for(i=0; i < count; i++ ) {
                                       tmp = tmp_adoubles[i];
                                       for(j=1; j< size ; j++)
                                          if ( tmp <= tmp_adoubles[j*count+i] )
                                             tmp = tmp_adoubles[j*count+i];
                                       rec_buf[i] = tmp;
                                   }
                                   break;
               case ADTL_MPI_MIN: for(i=0; i < count; i++ ) {
                                      tmp = tmp_adoubles[i];
                                      for(j=1; j< size ; j++)
                                         if ( tmp >= tmp_adoubles[j*count+i] )
                                            tmp = tmp_adoubles[j*count+i];
                                      rec_buf[i] = tmp;
                                   }
                                   break;
               case ADTL_MPI_SUM: for(i=0; i < count; i++ ) {
                                      tmp =0.;
                                      for(j=0; j< size ; j++){
                                         tmp += tmp_adoubles[j*count+i];
                                         }
                                       rec_buf[i] = tmp;
                                   }
                                   break;
               case ADTL_MPI_PROD:for(i=0; i < count; i++ ) {
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
    return ierr;
}

bool same_elem( unsigned int first, unsigned int second){
     return ( first == second );
}

int ADTL_MPI_get_sparse_pattern( adouble *a,unsigned int count, ADTL_MPI_Datatype type,
    int root, ADTL_MPI_Comm comm, unsigned int **&pat){
    int id, p_size, ierr = -1;
    MPI_Comm_size(comm, &p_size);
    MPI_Comm_rank(comm, &id);
    if( NULL != pat)
       free (pat);
    if( id == root)
       pat = (unsigned int**) malloc(count*sizeof(unsigned int*));
    unsigned int i,j;
    for( i=0; i < count ; i++){
       adouble b = a[i];
       unsigned int max_size =0 , real_size;
       unsigned int *sendbuf =NULL, *rec_buf =NULL, *tmp = NULL;
       tmp = (unsigned int*) malloc( sizeof(unsigned int)*p_size );
       real_size = (unsigned int) b.get_pattern_size();
       ierr = MPI_Allgather(&real_size,1,MPI_UNSIGNED, tmp , 1,MPI_UNSIGNED, comm);
       for( j=0; j < p_size ; j++)
          if( tmp[j] > max_size)
             max_size = tmp[j];

       if( max_size > 0 ) {
          sendbuf = (unsigned int*) malloc( max_size*sizeof(unsigned int) );
          const list<unsigned int>& tmp_set = b.get_pattern();
          list<unsigned int>::const_iterator it;
          sendbuf[0] = real_size;
          j=0;
          for(it = tmp_set.begin() ; it != tmp_set.end() ; it++,j++)
             sendbuf[j] = *it;
          if(id == root)
             rec_buf = (unsigned int*) malloc( max_size*p_size*sizeof(unsigned int) );

          ierr = MPI_Gather(sendbuf,max_size, MPI_UNSIGNED, rec_buf , max_size, MPI_UNSIGNED, root, comm);
          if(id == root){
             list<unsigned int> sp;
             for(j=0; j < p_size ; j++)
                 for(unsigned int l=0; l < tmp[j]; l++)
                    sp.push_back(  rec_buf[j*max_size+l]);
            sp.unique(same_elem);
            sp.sort();

            pat[i] = (unsigned int*) malloc(sizeof(unsigned int) * (sp.size() +1) );
            pat[i][0] = sp.size();
            unsigned int l=1;
            for(it = sp.begin() ; it != sp.end() ; it++,l++)
                pat[i][l] = *it;
            sp.clear();
          }
       } else {
          if( id == root) {
             pat[i] = (unsigned int*) malloc(sizeof(unsigned int));
             pat[i][0] =0;
          }
       }
    MPI_Barrier(comm);
    }

    return ierr;
}

}
/* That's all*/
