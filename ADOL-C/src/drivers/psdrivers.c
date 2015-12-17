/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     drivers/psdrivers.c
 Revision: $Id$
 Contents: Easy to use drivers for piecewise smooth functions
           (with C and C++ callable interfaces including Fortran 
            callable versions).

 Copyright (c) Andrea Walther, Sabrina Fiege 

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduct ion, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
  
----------------------------------------------------------------------------*/
#include <adolc/drivers/psdrivers.h>
#include <adolc/interfaces.h>
#include <adolc/adalloc.h>
#include "taping_p.h"
#include "dvlparms.h"

#include <math.h>


BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */

int abs_normal(short tag,      /* tape identifier */ 
               int m,          /* number od dependents   */             
               int n,          /* number of independents */
               int swchk,      /* number of switches (check) */
               double *x,      /* base point */ 
               short *sigma,   /* sigma of x */
               double *y,      /* function value */
               double *z,      /* switching variables */
               double *cz,     /* first constant */
               double *cy,     /* second constant */
               double **J,      
               double **Y,
               double **Z, 
               double **L)
{

  int i,j,s;
  double *res, tmp;
  s=get_num_switches(tag);
  
  /* This check is required because the user is probably allocating his 
   * arrays sigma, cz, Z, L, Y, J according to swchk */
  if (s != swchk) {
      fprintf(DIAG_OUT, "ADOL-C error: Number of switches passed %d does not "
              "match the one recorded on tape %d (%zu)\n", swchk, tag, s);
      adolc_exit(-1,"",__func__,__FILE__,__LINE__);
  }

  res=(double*)myalloc1(n+s);

  zos_pl_forward(tag,m,n,1,x,y,z);
  for(i=0;i<m+s;i++){
    int l = i - s;
    fos_pl_reverse(tag,m,n,s,i,res);
    if ( l < 0 ) {
        cz[i]=z[i];
        for(j=0;j<n;j++){
            Z[i][j]=res[j];
        }
        for(j=0;j<s;j++) { /* L[i][i] .. L[i][s] are theoretically zero,
                            *  we probably don't need to copy them */
            L[i][j]=res[j+n];	
            if (j < i)
	      {
		cz[i] = cz[i]-L[i][j]*fabs(z[j]);
	      }
        }
    } else {
        cy[l]=y[l];
        for(j=0;j<n;j++){
            J[l][j]=res[j];	
        }
        for(j=0;j<s;j++){
            Y[l][j]=res[j+n];
            cy[l] = cy[l]-Y[l][j]*fabs(z[j]);
        }
    }
  }

  myfree1(res);
  return 0;
}


END_C_DECLS
