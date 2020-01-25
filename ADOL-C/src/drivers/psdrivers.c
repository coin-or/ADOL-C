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
               double *y,      /* function value */
               double *z,      /* switching variables */
               double *cz,     /* first constant */
               double *cy,     /* second constant */
               double **Y,     /* m times n */
               double **J,     /* m times s */
               double **Z,     /* s times n */
               double **L)     /* s times s (lowtri) */
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
            Y[l][j]=res[j];
        }
        for(j=0;j<s;j++){
            J[l][j]=res[j+n];
            cy[l] = cy[l]-J[l][j]*fabs(z[j]);
        }
    }
  }

  myfree1(res);
  return 0;
}


/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient */
/*                                                                          */
int directional_active_gradient(short tag,      /* trace identifier */
                                int n,          /* number of independents */
                                double* x,      /* value of independents */
                                double* d,      /* direction */
                                double* g,      /* directional active gradient */
                                short *sigma_g  /* sigma of g */
                                )
{
  int i, j, p, k, s, max_dk, done, sum, keep;
  double max_entry, y, by;
  double *z;
  double **E, **invE, **grad, **gradu;

  keep = 1;
  by = 1;

  s=get_num_switches(tag);

  z = myalloc1(s);

  grad = (double**)myalloc2(1,n);
  gradu = (double**)myalloc2(s,n);
  E = (double**)myalloc2(n,n);

#if !defined(ADOLC_USE_CALLOC)
  memset(&(E[0][0]), 0, n*n*sizeof(double));
#endif

  max_dk=0;
  max_entry = -1;
  for(i=0;i<n;i++){
    E[i][0] = d[i];
    if(max_entry<fabs(d[i])){
      max_dk=i;
      max_entry = fabs(d[i]);
    }
  }

  k = 1; done = 0;
  j = 0;

  while((k<6) && (done == 0))
    {
      fov_pl_forward(tag,1,n,k,x,E,&y,grad,z,gradu,sigma_g);

      sum = 0;
      for(i=0;i<s;i++)
        {
          sum += fabs(sigma_g[i]);
        }

       if (sum == s)
        {

          zos_pl_forward(tag,1,n,keep,x,&y,z);
          fos_pl_sig_reverse(tag,1,n,s,sigma_g, &by ,g);
          done = 1;
        }
      else
        {
          if(j==max_dk)
            j++;
          E[j][k]=1;
          j++;
          k++;
        }
    }

  myfree1(z); myfree2(E); myfree2(grad); myfree2(gradu);

  if (done == 0)
    {
      fprintf(DIAG_OUT," NOT ENOUGH DIRECTIONS !!!!\n");
      adolc_exit(-1,"",__func__,__FILE__,__LINE__);
    }

  return 0;
}
