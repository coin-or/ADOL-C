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

#include <math.h>


BEGIN_C_DECLS

/****************************************************************************/
/*                                                 DRIVERS FOR PS FUNCTIONS */

void *mycalloc2(size_t n, size_t m, size_t size){
	uint8_t *temp;
	uint8_t **res;
	size_t i;

	temp=(uint8_t *)calloc(n*m*size+n,sizeof(void*));
	res=(uint8_t **)temp;
	temp+=n*sizeof(void*);
	for(i=0;i<n;i++){
		res[i]=temp;
		temp+=m*size;
	}
	return res;
}


/*--------------------------------------------------------------------------*/
/*                                              directional_active_gradient */
/*                                                                          */
int directional_active_gradient(short tag,      /* trace identifier */
				int n,          /* number of independents */
				double* x,      /* value of independents */
				short *sigma_x, /* sigma of x */
				double* d,      /* direction */
				double* g,      /* directional active gradient */
				short *sigma_g  /* sigma of g */
				)
{
  int i, j, p, k, s, max_dk, done, sum;
  double max_entry, y;
  double *z;
  double **E, **invE, **grad, **gradu;

  s=get_num_switches(tag);
  
  z = myalloc1(s);

  grad = (double**)mycalloc2(1,n,sizeof(double));
  gradu = (double**)mycalloc2(s,n,sizeof(double));
  E = (double**)mycalloc2(n,n,sizeof(double));


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
      fov_pl_sig_forward(tag,1,n,n-1,x,E,s,sigma_x,NULL,&y,grad,z,gradu,sigma_g);

      printf(" sigma_g \n");
      sum = 0;
      for(i=0;i<s;i++)
	{
	  printf(" %d ",sigma_g[i]);
	  printf("\n");
	  sum += fabs(sigma_g[i]);
	}
      
      if (sum == s)
	{
	  printf(" call reverse routine \n");
	  done = 1;
	}
      else
	{
	  if(j==max_dk)
	    j++;
	  E[j][k]=1;
	  j++;
	  k++;
	  printf(" hier \n");
	}
    }  

  myfree1(z); free(E); free(grad); free(gradu);

}

/*--------------------------------------------------------------------------*/
/*                                                          abs-normal form */

int abs_normal(short tag,      /* tape identifier */ 
               int m,          /* number od dependents   */             
               int n,          /* number of independents */
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
  double **res, tmp;
  s=get_num_switches(tag);

  res=(double**)myalloc2(m+s,n+s);

  zos_pl_forward(tag,m,n,1,x,y,z);
  for(i=0;i<m+s;i++){
    fos_pl_reverse(tag,m,n,s,i,res[i]);
  }

  for(i=0;i<s;i++){
    cz[i]=z[i];
    for(j=0;j<n;j++){
      Z[i][j]=res[i][j];
    }
    for(j=0;j<s;j++){
      L[i][j]=res[i][j+n];	
    }
    for(j=0;j<i;j++){
      cz[i] = cz[i]-L[i][j]*sigma[j]*z[j];	
    }
  }

  for(i=0;i<m;i++){
    cy[i]=y[i];
    for(j=0;j<n;j++){
      J[i][j]=res[i+s][j];	
    }
    for(j=0;j<s;j++){
      Y[i][j]=res[i+s][j+n];	
      cy[i] = cy[i]-Y[i][j]*sigma[j]*z[j];	
    }
  }

  myfree2(res);
}


END_C_DECLS
