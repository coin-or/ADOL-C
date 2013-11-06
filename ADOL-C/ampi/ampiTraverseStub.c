/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     ampiTraverseStub.c
 Revision: $Id$
 Contents: forward and reverse traversal stub for extra functionality
 
 Copyright (c) Kshitij Kulshreshtha

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#include "taping_p.h"
#include "traverse_stub.h"
#include "ampisupportAdolc.h"
#include "ampi/tape/support.h"

#include <stdio.h>

static int ampi_forward_function(enum OPCODES operation) {
    MPI_Op op;
    void *buf, *rbuf;
    int count, rcount;
    MPI_Datatype datatype, rtype;
    int src;
    int tag;
    enum AMPI_PairedWith_E pairedWith;
    MPI_Comm comm;
    MPI_Status* status;
    struct AMPI_Request_S request;
    switch(operation) {
	/*--------------------------------------------------------------------------*/
    case ampi_send: {
	ADOLC_TLM_AMPI_Send(buf,
			    count,
			    datatype,
			    src,
			    tag,
			    pairedWith,
			    comm);
	break;
    }
    case ampi_recv: {
	ADOLC_TLM_AMPI_Recv(buf,
                            count,
                            datatype,
                            src,
                            tag,
                            pairedWith,
                            comm,
                            status);
	break;
    }
    case ampi_isend: {
	ADOLC_TLM_AMPI_Isend(buf,
			     count,
			     datatype,
			     src,
			     tag,
			     pairedWith,
			     comm,
			     &request);
	break;
    }
    case ampi_irecv: {
	ADOLC_TLM_AMPI_Irecv(buf,
			     count,
			     datatype,
			     src,
			     tag,
			     pairedWith,
			     comm,
			     &request);
	break;
    }
    case ampi_wait: {
	ADOLC_TLM_AMPI_Wait(&request,
			    status);
	break;
    }
    case ampi_barrier: {
	ADOLC_TLM_AMPI_Barrier(comm);
	break;
    }
    case ampi_gather: {
	ADOLC_TLM_AMPI_Gather(buf,
			      count,
			      datatype,
			      rbuf,
			      rcount,
			      rtype,
			      src,
			      comm);
	break;
    }
    case ampi_scatter: {
	ADOLC_TLM_AMPI_Scatter(rbuf,
			       rcount,
			       rtype,
			       buf,
			       count,
			       datatype,
			       src,
			       comm);
	break;
    }
    case ampi_allgather: {
	ADOLC_TLM_AMPI_Allgather(buf,
				 count,
				 datatype,
				 rbuf,
				 rcount,
				 rtype,
				 comm);
	break;
    }
    case ampi_gatherv: {
	ADOLC_TLM_AMPI_Gatherv(buf,
			       count,
			       datatype,
			       rbuf,
			       NULL,
			       NULL,
			       rtype,
			       src,
			       comm);
	break;
    }
    case ampi_scatterv: {
	ADOLC_TLM_AMPI_Scatterv(rbuf,
				NULL,
				NULL,
				rtype,
				buf,
				count,
				datatype,
				src,
				comm);
	break;
    }
    case ampi_allgatherv: {
	ADOLC_TLM_AMPI_Allgatherv(buf,
				  count,
				  datatype,
				  rbuf,
				  NULL,
				  NULL,
				  rtype,
				  comm);
	break;
    }
    case ampi_bcast: {
	ADOLC_TLM_AMPI_Bcast(buf,
			     count,
			     datatype,
			     src,
			     comm);
	break;
    }
    case ampi_reduce: {
	ADOLC_TLM_AMPI_Reduce(buf,
			      rbuf,
			      count,
			      datatype,
			      op,
			      src,
			      comm);
	break;
    }
    case ampi_allreduce: {
	ADOLC_TLM_AMPI_Allreduce(buf,
				 rbuf,
				 count,
				 datatype,
				 op,
				 comm);
	break;
    }
    default:
	return 0;
    }
    return 1;
}

static int ampi_reverse_function(enum OPCODES operation) {
    MPI_Op op;
    void *buf, *rbuf;
    int count, rcount;
    MPI_Datatype datatype, rtype;
    int src; 
    int tag;
    enum AMPI_PairedWith_E pairedWith;
    MPI_Comm comm;
    MPI_Status* status;
    struct AMPI_Request_S request;
    switch(operation) {
                /*--------------------------------------------------------------------------*/
            case ampi_send: {
              BW_AMPI_Send(buf,
                           count,
                           datatype,
                           src,
                           tag,
                           pairedWith,
                           comm);
              break;
            }
            case ampi_recv: {
	      BW_AMPI_Recv(buf,
			   count,
			   datatype,
			   src,
			   tag,
			   pairedWith,
			   comm,
			   status);
	      break;
	    }
	  case ampi_isend: { 
	    BW_AMPI_Isend(buf,
			  count,
			  datatype,
			  src,
			  tag,
			  pairedWith,
			  comm,
			  &request);
	    break;
	  }
          case ampi_irecv: {
            BW_AMPI_Irecv(buf,
                          count,
                          datatype,
                          src,
                          tag,
                          pairedWith,
                          comm,
                          &request);
            break;
          }
	  case ampi_wait: { 
	    BW_AMPI_Wait(&request,
			 status);
	    break;
	  }
	  case ampi_barrier: {
	    BW_AMPI_Barrier(comm);
	    break;
	  }
	  case ampi_gather: { 
	    BW_AMPI_Gather(buf,
			   count,
			   datatype,
			   rbuf,
			   rcount,
			   rtype,
			   src,
			   comm);
	    break;
	  }
	  case ampi_scatter: {
	    BW_AMPI_Scatter(rbuf,
			    rcount,
			    rtype,
			    buf,
			    count,
			    datatype,
			    src,
			    comm);
	    break;
	  }
	  case ampi_allgather: {
	    BW_AMPI_Allgather(buf,
	                      count,
	                      datatype,
	                      rbuf,
	                      rcount,
	                      rtype,
	                      comm);
	    break;
	  }
	  case ampi_gatherv: {
	    BW_AMPI_Gatherv(buf,
			    count,
			    datatype,
			    rbuf,
			    NULL,
			    NULL,
			    rtype,
			    src,
			    comm);
	    break;
	  }
	  case ampi_scatterv: { 
	    BW_AMPI_Scatterv(rbuf,
			     NULL,
			     NULL,
			     rtype,
			     buf,
			     count,
			     datatype,
			     src,
			     comm);
	    break;
	  }
	  case ampi_allgatherv: {
	    BW_AMPI_Allgatherv(buf,
	                       count,
	                       datatype,
	                       rbuf,
	                       NULL,
	                       NULL,
	                       rtype,
	                       comm);
	    break;
	  }
	  case ampi_bcast: {
	    BW_AMPI_Bcast(buf,
			  count,
			  datatype,
			  src,
			  comm);
	    break;
	  }
	  case ampi_reduce: {
	    BWB_AMPI_Reduce(buf,
			   rbuf,
			   count,
			   datatype,
			   op,
			   src,
			   comm);
	    break;
	  }
	  case ampi_allreduce: {
	    BW_AMPI_Allreduce(buf,
	                      rbuf,
	                      count,
	                      datatype,
	                      op,
	                      comm);
	    break;
	  }
    default:
	return 0;
    }
    return 1;
}

extern void filewrite_ampi( unsigned short opcode, const char* opString, int nloc, int *loc);

static int ampi_tapedoc_function(enum OPCODES operation) {
    locint size;
    int l;
    int loc_a[maxLocsPerOp];
    MPI_Datatype anMPI_Datatype;
    MPI_Comm anMPI_Comm;
    MPI_Request anMPI_Request;
    MPI_Op anMPI_Op;
    int i;
    double aDouble;
    switch(operation) {
            case ampi_send:
	        loc_a[0] = get_locint_f();   /* start loc */
	        TAPE_AMPI_read_int(loc_a+1); /* count */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	        TAPE_AMPI_read_int(loc_a+2); /* endpoint */
	        TAPE_AMPI_read_int(loc_a+3); /* tag */
	        TAPE_AMPI_read_int(loc_a+4); /* pairedWith */
	        TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
		filewrite_ampi(operation, "ampi send",5, loc_a);
		break; 

            case ampi_recv:
                loc_a[0] = get_locint_f();   /* start loc */
                TAPE_AMPI_read_int(loc_a+1); /* count */
                TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
                TAPE_AMPI_read_int(loc_a+2); /* endpoint */
                TAPE_AMPI_read_int(loc_a+3); /* tag */
                TAPE_AMPI_read_int(loc_a+4); /* pairedWith */
                TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
                filewrite_ampi(operation, "ampi recv",5, loc_a);
                break;

            case ampi_isend: 
              /* push is delayed to the accompanying completion */
              TAPE_AMPI_read_MPI_Request(&anMPI_Request);
              filewrite_ampi(operation, "ampi isend",0, loc_a);
              break;

            case ampi_irecv:
              /* push is delayed to the accompanying completion */
              TAPE_AMPI_read_MPI_Request(&anMPI_Request);
              filewrite_ampi(operation, "ampi irecv",0, loc_a);
              break;

            case ampi_wait: 
	      /* for the operation we had been waiting for */
              size=0;
              loc_a[size++] = get_locint_f(); /* start loc */
              TAPE_AMPI_read_int(loc_a+size++); /* count */
              TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
              TAPE_AMPI_read_int(loc_a+size++); /* endpoint */
              TAPE_AMPI_read_int(loc_a+size++); /* tag */
              TAPE_AMPI_read_int(loc_a+size++); /* pairedWith */
              TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
              TAPE_AMPI_read_MPI_Request(&anMPI_Request);
              TAPE_AMPI_read_int(loc_a+size++); /* origin */
              filewrite_ampi(operation, "ampi wait",size, loc_a);
              break;

            case ampi_barrier:
              TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
              filewrite_ampi(operation, "ampi barrier",0, loc_a);
              break;

	    case ampi_bcast:
	      loc_a[0] = get_locint_f();   /* start loc */
	      TAPE_AMPI_read_int(loc_a+1); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	      TAPE_AMPI_read_int(loc_a+2); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      filewrite_ampi(operation, "ampi bcast",3, loc_a);
	      break;

	    case ampi_reduce:
	      loc_a[0] = get_locint_f();   /* rbuf */
	      loc_a[1] = get_locint_f();   /* sbuf */
	      TAPE_AMPI_read_int(loc_a+2); /* count */
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData */
	      i=0; /* read stored double array into dummy variable */
	      while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      if (loc_a[3]) {
	        i=0; /* for root, also read stored reduction result */
	        while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      }
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData again */
	      TAPE_AMPI_read_MPI_Op(&anMPI_Op);
	      TAPE_AMPI_read_int(loc_a+4); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	      TAPE_AMPI_read_int(loc_a+2); /* count again */
	      filewrite_ampi(operation, "ampi reduce",5, loc_a);
	      break;

	    case ampi_allreduce:
	      loc_a[0] = get_locint_f();   /* rbuf */
	      loc_a[1] = get_locint_f();   /* sbuf */
	      TAPE_AMPI_read_int(loc_a+2); /* count */
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData */
	      i=0; /* read off stored double array into dummy variable */
	      while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      if (loc_a[3]) {
	        i=0; /* for root, also read off stored reduction result */
	        while (i<loc_a[2]) { TAPE_AMPI_read_double(&aDouble); i++; }
	      }
	      TAPE_AMPI_read_int(loc_a+3); /* pushResultData again */
	      TAPE_AMPI_read_MPI_Op(&anMPI_Op);
	      TAPE_AMPI_read_int(loc_a+4); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype);
	      TAPE_AMPI_read_int(loc_a+2); /* count again */
	      filewrite_ampi(operation, "ampi allreduce",5, loc_a);
	      break;

	    case ampi_gather:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      if (*(loc_a+0)>0) {
	        loc_a[size++] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_int(loc_a+size++); /* rcnt */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
	      loc_a[size++]=get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi gather",size, loc_a);
	      break;

	    case ampi_scatter:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      if (*(loc_a+0)>0) {
	        loc_a[size++] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_int(loc_a+size++); /* rcnt */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
	      loc_a[size++]=get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi scatter",size, loc_a);
	      break;

	    case ampi_allgather:
	      TAPE_AMPI_read_int(loc_a+1); /* commSizeForRootOrNull */
	      if (*(loc_a+1)>0) {
	        TAPE_AMPI_read_int(loc_a+2); /* rcnt */
	        loc_a[2] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
	      TAPE_AMPI_read_int(loc_a+3); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+1); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi allgather",4, loc_a);
	      break;

	    case ampi_gatherv:
	      size=0;
	      TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
	      if (*(loc_a+0)>0) {
	        loc_a[size++] = get_locint_f(); /* rbuf loc */
	        TAPE_AMPI_read_int(loc_a+size++); /* rcnt[0] */
	        TAPE_AMPI_read_int(loc_a+size++); /* displs[0] */
	      }
	      for (l=1;l<*(loc_a+0);++l) {
	        TAPE_AMPI_read_int(loc_a+size);
	        TAPE_AMPI_read_int(loc_a+size);
	      }
	      if (*(loc_a+0)>0) {
	        TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
	      }
              loc_a[size++] = get_locint_f(); /* buf loc */
	      TAPE_AMPI_read_int(loc_a+size++); /* count */
	      TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
	      TAPE_AMPI_read_int(loc_a+size++); /* root */
	      TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
	      TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
	      filewrite_ampi(operation, "ampi gatherv",size, loc_a);
		break;

            case ampi_scatterv: 
              size=0;
              TAPE_AMPI_read_int(loc_a+size++); /* commSizeForRootOrNull */
              if (*(loc_a+0)>0) {
                loc_a[size++] = get_locint_f(); /* rbuf loc */
                TAPE_AMPI_read_int(loc_a+size++); /* rcnt[0] */
                TAPE_AMPI_read_int(loc_a+size++); /* displs[0] */
              }
              for (l=1;l<*(loc_a+0);++l) {
                TAPE_AMPI_read_int(loc_a+size);
                TAPE_AMPI_read_int(loc_a+size);
              }
              if (*(loc_a+0)>0) {
                TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
              }
              loc_a[size++] = get_locint_f(); /* buf loc */
              TAPE_AMPI_read_int(loc_a+size++); /* count */
              TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
              TAPE_AMPI_read_int(loc_a+size++); /* root */
              TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
              TAPE_AMPI_read_int(loc_a+0); /* commSizeForRootOrNull */
              filewrite_ampi(operation, "ampi scatterv",size, loc_a);
              break;

            case ampi_allgatherv:
                TAPE_AMPI_read_int(loc_a+1); /* commSizeForRootOrNull */
                for (l=0;l<*(loc_a+1);++l) {
                  TAPE_AMPI_read_int(loc_a+2); /* rcnts */
                  TAPE_AMPI_read_int(loc_a+2); /* displs */
                }
                if (*(loc_a+1)>0) {
                  loc_a[2] = get_locint_f(); /* rbuf loc */
                  TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* rtype */
                }
                TAPE_AMPI_read_int(loc_a+3); /* count */
                TAPE_AMPI_read_MPI_Datatype(&anMPI_Datatype); /* type */
                TAPE_AMPI_read_MPI_Comm(&anMPI_Comm);
                TAPE_AMPI_read_int(loc_a+1); /* commSizeForRootOrNull */
                filewrite_ampi(operation, "ampi allgatherv",4, loc_a);
                break;
    default:
	return 0;
    }
    return 1;
}

void ADOLC_AMPI_setup_stubs() {
    ampi_plugin = (ampi_traverse_plugin*)malloc(sizeof(ampi_traverse_plugin));
    ampi_plugin->forward_func = ampi_forward_function;
    ampi_plugin->reverse_func = ampi_reverse_function;
    ampi_plugin->tapedoc_func = ampi_tapedoc_function;
    ampi_plugin->init_for = TAPE_AMPI_resetBottom;
    ampi_plugin->init_rev = TAPE_AMPI_resetTop;
}
