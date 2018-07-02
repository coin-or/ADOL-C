#if !defined(ADOLC_DRIVERS_PARDRIVERS_H)
#define ADOLC_DRIVERS_PARDRIVERS_H 1

#include <adolc/internal/common.h>

/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                             par_jacobian */
/* par_jacobian(tag, m, n, x[n], J[m][n])                                   */
ADOLC_DLL_EXPORT int par_jacobian(short,int,int,const double*,double**);
ADOLC_DLL_EXPORT int par_jac_mat(short , int, int, int, const double*,
                                 double**, double**);

ADOLC_DLL_EXPORT int par_mat_jac(short , int, int, int, const double*,
                                 double**, double**);

/****************************************************************************/
#endif
