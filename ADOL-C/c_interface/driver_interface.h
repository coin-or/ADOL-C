/*
This file generates an interface to the overloaded forward and reverse mode
calls.
*/

#ifndef DRIVER_INTERFACE_H
#define DRIVER_INTERFACE_H

#ifdef __cplusplus
extern "C" {
#endif
/* forward(tag, m, n, d, keep, X[n][d+1], Y[m][d+1])                        */
int forward1(short tag, int m, int n, int d, int keep, double **X, double **Y);

/* forward(tag, m, n, d, keep, X[n][d+1], Y[d+1]) : hos || fos || zos       */
int forward2(short tag, int m, int n, int d, int keep, double **X, double *Y);

/* forward(tag, m, n, d, keep, X[n], Y[m]) : zos                            */
int forward3(short tag, int m, int n, int d, int keep, double *X, double *Y);

/* forward(tag, m, n, keep, X[n], Y[m]) : zos                               */
int forward4(short tag, int m, int n, int keep, double *X, double *Y);

/* forward(tag, m, n, d, p, x[n], X[n][p][d], y[m], Y[m][p][d]) : hov       */
int forward5(short tag, int m, int n, int d, int p, double *x, double ***X,
             double *y, double ***Y);

/* forward(tag, m, n, p, x[n], X[n][p], y[m], Y[m][p]) : fov                */
int forward6(short tag, int m, int n, int p, double *x, double **X, double *y,
             double **Y);

/* reverse(tag, m, n, d, u[m], Z[n][d + 1]) */
int reverse1(short tag, int m, int n, int d, double *u, double **Z);

/* reverse(tag, 1, n, 0, u, Z[n][d+1]), m=1 => u scalar                     */
int reverse2(short tag, int m, int n, int d, double u, double **Z);

/* reverse(tag, m, n, 0, u[m], Z[n]), d=0                                   */
int reverse3(short tag, int m, int n, int d, double *u, double *Z);

/* reverse(tag, 1, n, 0, u, Z[n]), m=1 and d=0 => u and Z scalars           */
int reverse4(short tag, int m, int n, int d, double u, double *Z);

/* reverse(tag, m, n, d, q, U[q][m], Z[q][n][d+1], nz[q][n])          */
int reverse5(short tag, int m, int n, int d, int q, double **U, double ***Z,
             short **nz);

/* reverse(tag, m, n, d, q, U[q], Z[q][n][d+1], nz[q][n]) : hov             */
int reverse6(short tag, int m, int n, int d, int q, double *U, double ***Z,
             short **nz);

/* reverse(tag, 1, n, d, q, U[q], Z[q][n][d+1], nz[q][n]), m=1 => u vector  */
int reverse7(short tag, int m, int n, int d, int q, double **U, double **Z);

/* reverse(tag, m, n, q, U[q][m], Z[q][n])                          */
int reverse8(short tag, int m, int n, int q, double **U, double **Z);

/* reverse(tag, m, n, d, q, U[q], Z[q][n])                         */
int reverse9(short tag, int m, int n, int d, int q, double *U, double **Z);

/* reverse(tag, m, n, d, Z[q][n][d+1], nz[q][n]) : hov                      */
int reverse10(short tag, int m, int n, int d, double ***Z, short **nz);

#ifdef __cplusplus
}
#endif

#endif // DRIVER_INTERFACE_H