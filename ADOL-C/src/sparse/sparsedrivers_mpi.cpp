/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     sparse/sparsedrivers_mpi.cpp
 Revision: $Id$
 Contents: "Easy To Use" C++ parallel interfaces of SPARSE package

 Copyright (c) Benjamin Letschert

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/
#include <adolc/sparse/sparsedrivers.h>
#include <adolc/sparse/sparsedrivers_mpi.h>
#include <adolc/oplate.h>
#include <adolc/adalloc.h>
#include <adolc/interfaces_mpi.h>
#include "taping_p.h"

#if defined(ADOLC_INTERNAL)
#    if HAVE_CONFIG_H
#        include "config.h"
#    endif
#endif

#if HAVE_LIBCOLPACK
#include "ColPackHeaders.h"
#endif

#include <math.h>
#include <cstring>

#if HAVE_LIBCOLPACK
using namespace ColPack;
#endif

using namespace std;

/****************************************************************************/
/*******       sparse Jacobains, separate drivers             ***************/
/****************************************************************************/

#ifdef HAVE_MPI
#include <adolc/adolc_mpi.h>
static void deepcopy_HP(unsigned int ***HPnew, unsigned int **HP, int indep)
{
    int i,j,s;
    *HPnew = (unsigned int **)malloc(indep*sizeof(unsigned int *));
    for (i=0; i<indep; i++) {
       s=HP[i][0];
       (*HPnew)[i] = (unsigned int *)malloc((s+1)*(sizeof(unsigned int)));
       for (j=0; j<=s; j++)
           (*HPnew)[i][j] = HP[i][j];
    }
}


/* now the C - functions                */

int jac_pat_mpi(int id, int size, short tag, int depen,int indep,const double *basepoint, unsigned int **crs, int *options ){
     int rc= -1;
     int i, ctrl_options[2];

    if (crs == NULL) {
        fprintf(DIAG_OUT,"ADOL-C user error in jac_pat(...) : "
                "parameter crs may not be NULL !\n");
        exit(-1);
    } else
     if ( id==0){
        for (i=0; i<depen; i++)
            crs[i] = NULL;

        if (( options[0] < 0 ) || (options[0] > 1 ))
           options[0] = 0; /* default */
        if (( options[1] < 0 ) || (options[1] > 1 ))
           options[1] = 0; /* default */
        if (( options[2] < -1 ) || (options[2] > 2 ))
           options[2] = 0; /* default */
        if (options[0] == 0) {
           if (options[1] == 1)
             rc = indopro_forward_tight_mpi(id, size,tag, depen, indep, basepoint, crs);
           else
             rc = indopro_forward_safe_mpi(id,size,tag, depen, indep, basepoint, crs);
        } else {
          ctrl_options[0] = options[1];
          ctrl_options[1] = options[2];
          rc = bit_vector_propagation(size*tag+id , depen, indep,  basepoint, crs, ctrl_options);
        }
     } else {
        if (( options[0] < 0 ) || (options[0] > 1 ))
           options[0] = 0; /* default */
        if (( options[1] < 0 ) || (options[1] > 1 ))
           options[1] = 0; /* default */
        if (( options[2] < -1 ) || (options[2] > 2 ))
           options[2] = 0; /* default */

        if (options[1] == 1)
           rc = indopro_forward_tight_mpi(id,size,tag, 0, 0, NULL, crs);
        else
           rc = indopro_forward_safe_mpi(id,size,tag, 0, 0, NULL, crs);
     }
     return(rc);
}

int sparse_jac_mpi(int id,int size, short tag, int depen, int indep, int repeat,const double* basepoint, int *nnz, unsigned int **rind, unsigned int **cind, double **values, int *options)
{
#ifdef HAVE_LIBCOLPACK
     int this_tag = size*tag +id;
     int rc =-3, tmp;
     bool forward, tight;

     if (options[0] != 0){
        if (id==0)
           fprintf(DIAG_OUT,"ADOL-C error: Propagation of bit pattern not yet implemented.\n");
        exit(-1);
     }
     if (id==0){
        int i;
        unsigned int j;
        SparseJacInfos sJinfos;
        int dummy;
        BipartiteGraphPartialColoringInterface *g;
        TapeInfos *tapeInfos;
        JacobianRecovery1D *jr1d;
        JacobianRecovery1D jr1d_loc;

        ADOLC_OPENMP_THREAD_NUMBER;
        ADOLC_OPENMP_GET_THREAD_NUMBER;

        if (repeat == 0) {
           if (( options[0] < 0 ) || (options[0] > 1 ))
              options[0] = 0; /* default */
           if (( options[1] < 0 ) || (options[1] > 1 ))
              options[1] = 0; /* default */
           if (( options[2] < -1 ) || (options[2] > 2 ))
              options[2] = 0; /* default */
           if (( options[3] < 0 ) || (options[3] > 1 ))
              options[3] = 0; /* default */

           sJinfos.JP = (unsigned int **) malloc(depen*sizeof(unsigned int *));
           rc = jac_pat_mpi(id,size,tag, depen, indep, basepoint, sJinfos.JP, options);

           if (rc < 0) {
              fprintf(DIAG_OUT,"ADOL-C error in parallel sparse_jac() computing jac_pat()\n");
              exit(rc);
           }

           sJinfos.depen = depen;
           sJinfos.nnz_in = depen;
           sJinfos.nnz_in = 0;
           for (i=0;i<depen;i++) {
               for (j=1;j<=sJinfos.JP[i][0];j++)
                   sJinfos.nnz_in++;
           }

           *nnz = sJinfos.nnz_in;

           if (options[2] == -1){
              (*rind) = new unsigned int[*nnz];
              (*cind) = new unsigned int[*nnz];
              unsigned int index = 0;
              for (i=0;i<depen;i++)
                  for (j=1;j<=sJinfos.JP[i][0];j++){
                      (*rind)[index] = i;
                      (*cind)[index++] = sJinfos.JP[i][j];
                  }
           }

           /* sJinfos.Seed is memory managed by ColPack and will be deleted
            * along with g. We only keep it in sJinfos for the repeat != 0 case */

           g = new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, sJinfos.JP, depen, indep);
           jr1d = new JacobianRecovery1D;

           if (options[3] == 1) {
              g->GenerateSeedJacobian(&(sJinfos.Seed), &(sJinfos.seed_rows),&(sJinfos.seed_clms),  "SMALLEST_LAST","ROW_PARTIAL_DISTANCE_TWO");
              sJinfos.seed_clms = indep;
           }
           else
           {
              g->GenerateSeedJacobian(&(sJinfos.Seed), &(sJinfos.seed_rows),
                                &(sJinfos.seed_clms), "SMALLEST_LAST","COLUMN_PARTIAL_DISTANCE_TWO");
              sJinfos.seed_rows = depen;
           }

           sJinfos.B = myalloc2(sJinfos.seed_rows,sJinfos.seed_clms);
           sJinfos.y = myalloc1(depen);

           sJinfos.g = (void *) g;
           sJinfos.jr1d = (void *) jr1d;
           setTapeInfoJacSparse(this_tag, sJinfos);
           tapeInfos=getTapeInfos(this_tag);
           memcpy(&ADOLC_CURRENT_TAPE_INFOS, tapeInfos, sizeof(TapeInfos));
        }
        else
        {
           tapeInfos=getTapeInfos(this_tag);
           memcpy(&ADOLC_CURRENT_TAPE_INFOS, tapeInfos, sizeof(TapeInfos));
           sJinfos.depen    = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.depen;
           sJinfos.nnz_in    = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.nnz_in;
           sJinfos.JP        = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.JP;
           sJinfos.B         = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.B;
           sJinfos.y         = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.y;
           sJinfos.Seed      = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.Seed;
           sJinfos.seed_rows = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.seed_rows;
           sJinfos.seed_clms = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.seed_clms;
           g = (BipartiteGraphPartialColoringInterface *)ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.g;
           jr1d = (JacobianRecovery1D *)ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sJinfos.jr1d;
        }

        if (sJinfos.nnz_in != *nnz) {
           fprintf(DIAG_OUT," ADOL-C error in parallel sparse_jac():"
               " Number of nonzeros not consistent,"
               " repeat call with repeat = 0 \n");
           exit(-3);
        }

        if (options[2] == -1)
           return rc;

        // send count for depth
        if (options[3] == 1)
           tmp = sJinfos.seed_rows;
        else
           tmp = sJinfos.seed_clms;

//        for(i=1; i < size ; i++)
  //           MPI_Send(&tmp,1,MPI_INT,i,0,MPI_COMM_WORLD);
        MPI_Bcast(&tmp,1,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        /* compute jacobian times matrix product */
        if (options[3] == 1){
           rc = zos_forward_mpi(id,size,tag,depen,indep,1,basepoint,sJinfos.y);
           if (rc < 0)
              return rc;
           rc = fov_reverse_mpi(id,size,tag,depen,indep,sJinfos.seed_rows,sJinfos.Seed,sJinfos.B);
        }
        else
        {
           rc = fov_forward(this_tag, depen, indep, sJinfos.seed_clms, basepoint, sJinfos.Seed, sJinfos.y, sJinfos.B);
        }

        /* recover compressed Jacobian => ColPack library */

        if (*values != NULL && *rind != NULL && *cind != NULL) {
        // everything is preallocated, we assume correctly
        // call usermem versions
           if (options[3] == 1)
              jr1d->RecoverD2Row_CoordinateFormat_usermem(g, sJinfos.B, sJinfos.JP, rind, cind, values);
           else
              jr1d->RecoverD2Cln_CoordinateFormat_usermem(g, sJinfos.B, sJinfos.JP, rind, cind, values);
        } else {
      // at least one of rind cind values is not allocated, deallocate others
      // and call unmanaged versions
           if (*values != NULL)
              free(*values);
           if (*rind != NULL)
              free(*rind);
           if (*cind != NULL)
              free(*cind);
           if (options[3] == 1)
              jr1d->RecoverD2Row_CoordinateFormat_unmanaged(g, sJinfos.B, sJinfos.JP, rind, cind, values);
           else
              jr1d->RecoverD2Cln_CoordinateFormat_unmanaged(g, sJinfos.B, sJinfos.JP, rind, cind, values);
        }
     } // end of id == 0
     else {
          if (repeat == 0) {
             if (( options[0] < 0 ) || (options[0] > 1 ))
                options[0] = 0; /* default */
             if (( options[1] < 0 ) || (options[1] > 1 ))
                options[1] = 0; /* default */
             if (( options[2] < -1 ) || (options[2] > 2 ))
                options[2] = 0; /* default */
             if (( options[3] < 0 ) || (options[3] > 1 ))
                options[3] = 0; /* default */

             rc = jac_pat_mpi(id,size,tag, depen, indep,NULL,NULL, options);
             if (rc < 0) {
                  fprintf(DIAG_OUT," ADOL-C error in parallel sparse_jac() computing jac_pat()\n");
                  exit(rc);
             }
          } // end of sparse_jac doings

	  if (options[2] == -1)
		  return rc;

       MPI_Bcast(&tmp,1,MPI_INT,0,MPI_COMM_WORLD);
       MPI_Barrier(MPI_COMM_WORLD);


          if (options[3] == 1){
             rc = zos_forward_mpi(id,size,tag,0,0,1,NULL,NULL);
             if (rc < 0) {
		   fprintf(DIAG_OUT,"ADOL-C error in parallel sparse_jac() computing zos_forward()\n");
                   exit(rc);
             }
             rc = fov_reverse_mpi(id,size,tag,0,0,tmp,NULL,NULL);
          }
          else {
             rc = fov_forward_mpi(id,size,tag, 0,0, tmp, NULL, NULL, NULL, NULL);
          }
     } // end of ID != 0

     return rc;
#else
    fprintf(DIAG_OUT, "ADOL-C error: function %s can only be used if linked with ColPack\n", __FUNCTION__);
    exit(-1);
#endif
}

int hess_pat_mpi(
     int id,
     int size,
     short          tag,        /* tape identification                        */
     int            indep,      /* number of independent variables            */
     const double  *basepoint,  /* independant variable values                */
     unsigned int **crs,
     /* returned compressed row block-index storage                         */
     int          option
     /* control option
        option : test the computational graph control flow
                               0 - safe mode (default)
                               1 - tight mode                              */

) {
    int rc= -1;
     int i;

     if (id == 0) {
        if (crs == NULL) {
           fprintf(DIAG_OUT,"ADOL-C user error in parallel hess_pat(...) : "
                "parameter crs may not be NULL !\n");
           exit(-1);
        }else
           for (i=0; i<indep; i++)
               crs[i] = NULL;

        if (( option < 0 ) || (option > 2 ))
           option = 0;   /* default */

        if (option == 1)
           rc = nonl_ind_forward_tight_mpi(id,size,tag, 1, indep, basepoint, crs);
        else
           rc = nonl_ind_forward_safe_mpi(id,size,tag, 1, indep, basepoint, crs);
     } // end of id == 0
     else {
        if (( option < 0 ) || (option > 2 ))
           option = 0;   /* default */
        if (option == 1)
           rc = nonl_ind_forward_tight_mpi(id,size,tag, 0, 0, NULL, crs);
        else
           rc = nonl_ind_forward_safe_mpi(id,size,tag, 0, 0, NULL, crs);
     }
     return rc;
}

int sparse_hess_mpi(
     int id,
     int size,
     short          tag,        /* tape identification                     */
     int            indep,      /* number of independent variables         */
     int            repeat,     /* indicated repeated call with same seed  */
     const double  *basepoint,  /* independant variable values             */
     int           *nnz,        /* number of nonzeros                      */
     unsigned int **rind,       /* row index                               */
     unsigned int **cind,       /* column index                            */
     double       **values,     /* non-zero values                         */
     int           *options
     /* control options
                    options[0] :test the computational graph control flow
                               0 - safe mode (default)
                               1 - tight mode
                    options[1] : way of recovery
                               0 - indirect recovery
                               1 - direct recovery                         */
) {
#ifdef HAVE_LIBCOLPACK
     int i, l, tmp;
     int this_tag = size*tag + id;
     unsigned int j;
     SparseHessInfos sHinfos;
     double **Seed;
     int dummy;
     double y;
     int ret_val=-1;
     GraphColoringInterface *g;
     TapeInfos *tapeInfos;
     double *v, *w, **X, yt, lag=1;
     HessianRecovery *hr;

     ADOLC_OPENMP_THREAD_NUMBER;
     ADOLC_OPENMP_GET_THREAD_NUMBER;

     /* Generate sparsity pattern, determine nnz, allocate memory */
     if( id ==0){
        if (repeat <= 0) {
           if (( options[0] < 0 ) || (options[0] > 1 ))
              options[0] = 0; /* default */
           if (( options[1] < 0 ) || (options[1] > 1 ))
              options[1] = 0; /* default */

           if (repeat == 0) {
              sHinfos.HP    = (unsigned int **) malloc(indep*sizeof(unsigned int *));
              /* generate sparsity pattern */
              ret_val = hess_pat_mpi(id,size,tag, indep, basepoint, sHinfos.HP, options[0]);
              if (ret_val < 0) {
                 fprintf(DIAG_OUT," ADOL-C error in parallel sparse_hess() \n");
                 exit(ret_val);
              }
           }
           else {
              tapeInfos=getTapeInfos(this_tag);
              memcpy(&ADOLC_CURRENT_TAPE_INFOS, tapeInfos, sizeof(TapeInfos));
              if (indep != ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.indep) {
                 fprintf(DIAG_OUT,"ADOL-C Error: wrong number of independents stored in parallel hessian pattern.\n");
                 exit(-1);
              }
              deepcopy_HP(&sHinfos.HP,ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.HP,indep);
           }

           sHinfos.indep = indep;
           sHinfos.nnz_in = 0;

           for (i=0;i<indep;i++) {
               for (j=1;j<=sHinfos.HP[i][0];j++)
                   if ((int) sHinfos.HP[i][j] >= i)
                      sHinfos.nnz_in++;
           }

           *nnz = sHinfos.nnz_in;

           /* compute seed matrix => ColPack library */

           Seed = NULL;

           g = new GraphColoringInterface(SRC_MEM_ADOLC, sHinfos.HP, indep);
           hr = new HessianRecovery;

           if (options[1] == 0)
              g->GenerateSeedHessian(&Seed, &dummy, &sHinfos.p,"SMALLEST_LAST","ACYCLIC_FOR_INDIRECT_RECOVERY");
           else
              g->GenerateSeedHessian(&Seed, &dummy, &sHinfos.p,"SMALLEST_LAST","STAR");

           sHinfos.Hcomp = myalloc2(indep,sHinfos.p);
           sHinfos.Xppp = myalloc3(indep,sHinfos.p,1);

           for (i=0; i<indep; i++)
               for (l=0;l<sHinfos.p;l++)
                   sHinfos.Xppp[i][l][0] = Seed[i][l];

           /* Seed will be freed by ColPack when g is freed */
           Seed = NULL;

           sHinfos.Yppp = myalloc3(1,sHinfos.p,1);
           sHinfos.Zppp = myalloc3(sHinfos.p,indep,2);
           sHinfos.Upp = myalloc2(1,2);
           sHinfos.Upp[0][0] = 1;
           sHinfos.Upp[0][1] = 0;
           sHinfos.g = (void *) g;
           sHinfos.hr = (void *) hr;

           setTapeInfoHessSparse(this_tag, sHinfos);

           tapeInfos=getTapeInfos(this_tag);
           memcpy(&ADOLC_CURRENT_TAPE_INFOS, tapeInfos, sizeof(TapeInfos));
        } // end of repeat == 0
        else
        {
           tapeInfos=getTapeInfos(this_tag);
           memcpy(&ADOLC_CURRENT_TAPE_INFOS, tapeInfos, sizeof(TapeInfos));
           sHinfos.nnz_in = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.nnz_in;
           sHinfos.HP     = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.HP;
           sHinfos.Hcomp  = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.Hcomp;
           sHinfos.Xppp   = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.Xppp;
           sHinfos.Yppp   = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.Yppp;
           sHinfos.Zppp   = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.Zppp;
           sHinfos.Upp    = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.Upp;
           sHinfos.p      = ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.p;
           g = (GraphColoringInterface *)ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.g;
           hr = (HessianRecovery *)ADOLC_CURRENT_TAPE_INFOS.pTapeInfos.sHinfos.hr;
        }

        if (sHinfos.Upp == NULL) {
           fprintf(DIAG_OUT," ADOL-C error in parallel sparse_hess():\n First call with repeat = 0 \n");
           exit(-3);
        }

        if (sHinfos.nnz_in != *nnz) {
           fprintf(DIAG_OUT," ADOL-C error in parallel sparse_hess():\n Number of nonzeros not consistent,\n new call with repeat = 0 \n");
           exit(-3);
        }

        if (repeat == -1)
           return ret_val;

        v    = (double*) malloc(indep*sizeof(double));
        w    = (double*) malloc(indep*sizeof(double));
        X = myalloc2(indep,2);

        tmp = sHinfos.p;
        MPI_Bcast(&tmp,1, MPI_INT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        for (i = 0; i < sHinfos.p; ++i) {
            for (l = 0; l < indep; ++l)
                v[l] = sHinfos.Xppp[l][i][0];

            ret_val = fos_forward_mpi(id,size,tag, 1, indep, 2, basepoint, v, &y, &yt);
            MINDEC(ret_val, hos_reverse_mpi(id,size,tag, 1, indep, 1, &lag, X));
            for (l = 0; l < indep; ++l)
                sHinfos.Hcomp[l][i] = X[l][1];
        }

        myfree1(v);
        myfree1(w);
        myfree2(X);

        if (*values != NULL && *rind != NULL && *cind != NULL) {
           // everything is preallocated, we assume correctly
           // call usermem versions
           if (options[1] == 0)
              hr->IndirectRecover_CoordinateFormat_usermem(g, sHinfos.Hcomp, sHinfos.HP, rind, cind, values);
           else
              hr->DirectRecover_CoordinateFormat_usermem(g, sHinfos.Hcomp, sHinfos.HP, rind, cind, values);
        } else {
           // at least one of rind cind values is not allocated, deallocate others
           // and call unmanaged versions
          if (*values != NULL)
             free(*values);
          if (*rind != NULL)
             free(*rind);
          if (*cind != NULL)
             free(*cind);

          if (options[1] == 0)
             hr->IndirectRecover_CoordinateFormat_unmanaged(g, sHinfos.Hcomp, sHinfos.HP, rind, cind, values);
          else
             hr->DirectRecover_CoordinateFormat_unmanaged(g, sHinfos.Hcomp, sHinfos.HP, rind, cind, values);
        }
     } // ende id == 0
     else {
        if (repeat <= 0) {
           if (( options[0] < 0 ) || (options[0] > 1 ))
              options[0] = 0; /* default */
           if (( options[1] < 0 ) || (options[1] > 1 ))
              options[1] = 0; /* default */

           if (repeat == 0){
              /* generate sparsity pattern */
              ret_val = hess_pat_mpi(id,size,tag, indep, basepoint, NULL, options[0]);
              if (ret_val < 0) {
                 fprintf(DIAG_OUT," ADOL-C error in parallel sparse_hess() \n");
                 exit(ret_val);
              }
           }
        }

        if (repeat == -1)
           return ret_val;

        MPI_Bcast(&tmp,1, MPI_INT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        for (i = 0; i < tmp; ++i) {
            ret_val = fos_forward_mpi(id,size,tag, 0, 0, 2, NULL, NULL, NULL, NULL);
            MINDEC(ret_val, hos_reverse_mpi(id,size,tag, 0, 0, 1, &lag, NULL ) );
        }
     } // end id else
     return ret_val;
#else
    fprintf(DIAG_OUT, "ADOL-C error: function %s can only be used if linked with ColPack\n", __FUNCTION__);
    exit(-1);
#endif
}


int jac_pat(int id, int size, short tag, int depen,int indep, double *basepoint, unsigned int **crs, int *options ){
     return jac_pat_mpi(id,size,tag, depen, indep, basepoint, crs, options);
}
int hess_pat( int id, int size, short tag, int indep, double *basepoint, unsigned int **crs, int option){
     return hess_pat_mpi( id, size, tag, indep, basepoint, crs, option);
}

int sparse_jac( int id,int size ,short tag, int depen, int indep, int repeat,const double *basepoint, int *nnz, unsigned int **rind,
    unsigned int **cind, double **values,int *options ){
     return sparse_jac_mpi(id,size,tag,depen,indep,repeat,basepoint,nnz,rind,cind,values,options);
}

int sparse_hess( int id, int size ,short tag ,int indep,int repeat,const double *basepoint, int *nnz ,unsigned int **rind, unsigned int **cind, double **values, int *options){
     return sparse_hess_mpi(id,size,tag,indep,repeat,basepoint,nnz,rind,cind,values,options);
}



#endif
