#include "driver_interface.h"
#include <adolc/adolc.h>
/*
This file generates an interface to the overloaded forward and reverse mode
calls.
*/

int forward1(short tag, int m, int n, int d, int keep, double **X, double **Y) {
  return forward(tag, m, n, d, keep, X, Y);
}
int forward2(short tag, int m, int n, int d, int keep, double **X, double *Y) {
  return forward(tag, m, n, d, keep, X, Y);
}
int forward3(short tag, int m, int n, int d, int keep, double *X, double *Y) {
  return forward(tag, m, n, d, keep, X, Y);
}
int forward4(short tag, int m, int n, int keep, double *X, double *Y) {
  return forward(tag, m, n, keep, X, Y);
}
int forward5(short tag, int m, int n, int d, int p, double *x, double ***X,
             double *y, double ***Y) {
  return forward(tag, m, n, d, p, x, X, y, Y);
}

int forward6(short tag, int m, int n, int p, double *x, double **X, double *y,
             double **Y) {
  return forward(tag, m, n, p, x, X, y, Y);
}
int reverse1(short tag, int m, int n, int d, double *u, double **Z) {
  return reverse(tag, m, n, d, u, Z);
}
int reverse2(short tag, int m, int n, int d, double u, double **Z) {
  return reverse(tag, m, n, d, u, Z);
}
int reverse3(short tag, int m, int n, int d, double *u, double *Z) {
  return reverse(tag, m, n, d, u, Z);
}
int reverse4(short tag, int m, int n, int d, double u, double *Z) {
  return reverse(tag, m, n, d, u, Z);
}
int reverse5(short tag, int m, int n, int d, int q, double **U, double ***Z,
             short **nz) {
  return reverse(tag, m, n, d, q, U, Z, nz);
}
int reverse6(short tag, int m, int n, int d, int q, double *U, double ***Z,
             short **nz) {
  return reverse(tag, m, n, d, q, U, Z, nz);
}

int reverse7(short tag, int m, int n, int d, int q, double **U, double **Z) {
  return reverse(tag, m, n, d, q, U, Z);
}

int reverse8(short tag, int m, int n, int q, double **U, double **Z) {
  return reverse(tag, m, n, q, U, Z);
}
int reverse9(short tag, int m, int n, int d, int q, double *U, double **Z) {
  return reverse(tag, m, n, d, q, U, Z);
}

int reverse10(short tag, int m, int n, int d, double ***Z, short **nz) {
  return reverse(tag, m, n, d, Z, nz);
}
