/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     LUsolve_MT.cpp
 Revision: $Id$
 Contents: Serves as example and test for
             * Computation of the determinant of a matrix
               by LU-decomposition of the system matrix without pivoting 
             * application of tapedoc to observe taping of
               the new op_codes for the elementary operations

                     y += x1 * x2;
                     y -= x1 * x2;
             * application of par_jacobian driver
             * handling several tapes while using par_jacobian

 Usage:
   OMP_NUM_THREADS=N ./LUsolve_MT [SIZE1 [, SIZE2 [, SIZE3 ...]]]

 First, the LU decomposition for the provided system sizes is traced. In a
 second step, the drivers jacobian and par_jacobian are called to obtain
 the Jacobian matrices. For each system size the matrices returned from
 jacobian and par_jacobian are compared.

 jac_mat and mat_jac are called. To test the correctness of the result, the
 obtained jacobian matrix from a driver jacobian() is multiplied from right
 and left, respectively.

 After doing so in ascending order for system sizes (calc_seq), the drivers
 are called in random order (calc_rand).


 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz, 
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel
  
 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes 
 recipient's acceptance of the terms of the accompanying license file.
 
---------------------------------------------------------------------------*/

/****************************************************************************/
/*                                                                 INCLUDES */
#include "LU.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>


// At least some const correctness.
typedef const double* const* constMat;

// Global counter to ensure, that every Trace has a unique tag.
static int tagCntr = 0;

class Problem
{
public:
  ~Problem() {
    myfree1(args);
    myfree1(x);
  }
  uint size = 0;
  int tag = -1;
  uint depen = 0;
  uint indep = 0;
  double* args = NULL ;
  double* x = NULL;
};


int calc_seq(const std::vector<uint>& sizes);
int calc_rand(const std::vector<uint>& sizes);
int compute_and_trace(Problem& p);
int get_Jacobian(Problem& p, double** J);
int apply_drivers(Problem& p);
int compute_JU(uint m, uint n, uint l, Problem& p, double** U, double** J);
int compute_UJ(uint m, uint n, uint l, Problem& p, double** U, double** J);
void mat_print(const std::string& name, const uint m, const uint n,
               constMat M);
void mat_fill_iplusj(const uint m, const uint n, double* const* M);
void mat_fill_rand(const uint m, const uint n, double* const* M);
void mat_mul(constMat A, constMat B, double* const* C, const uint m,
             const uint n, const uint l);
int compare_mats(const uint m, const uint n, constMat jac1,
                 const std::string& name1, constMat jac2,
                 const std::string& name2);
void usage()
{
  std::cout << "Usage: OMP_NUM_THREADS=NUMTHREADS ./LUsolve_MT [SIZE1 [, SIZE2 [, SIZE3 ...]]] \n";
}

/****************************************************************************/
/*                                                             MAIN PROGRAM */
/*--------------------------------------------------------------------------*/
int main(int argc, char* argv []) {
    int size;
    std::vector<uint> sizes;
    if (2 <= argc) {
      for (int i = 1; i < argc; ++i) {
       size = atoi(argv[i]);
       if (1 > size) {
         usage();
         return 1;
       }
       else
         sizes.push_back(size);
      }
    } else {
      usage();
      return 0;
    }

    // Since the tag numbering is identical to problem sizes, we remove duplicates.
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    /*------------------------------------------------------------------------*/
    /* Info */
    std::cout << "LINEAR SYSTEM SOLVING by LU-DECOMPOSITION (ADOL-C Example)\n\n";

    std::cout << "=========================================================\n";
    std::cout << " Evaluation in sequential order \n";
    std::cout << "=========================================================\n";
    calc_seq(sizes);

    //std::cout << "=========================================================\n";
    //std::cout << " Evaluation in alternate order \n";
    //std::cout << "=========================================================\n";
    //calc_rand(sizes);

    return 0;
}

int calc_rand(const std::vector<uint>& sizes) {
  int ret = 0;
  std::vector<Problem> problems(sizes.size());
  std::vector<uint> shuffledSizes = sizes;
  std::random_shuffle(shuffledSizes.begin(), shuffledSizes.end());

  for (uint i = 0; i < shuffledSizes.size(); ++i) {
    std::cout << "=== System size is: " << shuffledSizes[i] << "\n";
    problems[i] = Problem{shuffledSizes[i], tagCntr++};
    compute_and_trace(problems[i]);
    apply_drivers(problems[i]);
    std::cout << "\n";
  }

  return ret;
}

int calc_seq(const std::vector<uint>& sizes) {
  int ret = 0;
  std::vector<Problem> problems(sizes.size());

  for (uint i = 0; i < sizes.size(); ++i) {
    std::cout << "=== System size is: " << sizes[i] << "\n";
    problems[i] = Problem{sizes[i], tagCntr++};
    compute_and_trace(problems[i]);
    apply_drivers(problems[i]);
    std::cout << "\n";
  }

  return ret;
}

/* [in]   size, tag
 * [out]  indep, depen, tag
 */
int compute_and_trace(Problem& p) {
  int ret = 0;

  // Variables
  int size = p.size;
  if (0 >= size)
    return 1;

  p.tag   = size;                    // tape tag
  p.indep = size*size+size;          // # of indeps
  p.depen = size;                    // # of deps

  // Passive variables
  double** A = myalloc2(size, size);
  double* a1 = myalloc1(size);
  double* a2 = myalloc1(size);
  double* b = myalloc1(size);
  p.x = myalloc1(size);
  adouble **AA, *AAp, *Abx;         // active variables
  p.args = myalloc1(p.indep);         // arguments


  /*------------------------------------------------------------------------*/
  /* Allocation and initialization of the system matrix */
  AA  = new adouble*[size];
  AAp = new adouble[size*size];
  for (int i = 0; i < size; ++i) {
    AA[i] = AAp;
    AAp += size;
  }

  Abx = new adouble[size];
  for (int i = 0; i < size; ++i) {
    a1[i] = i*0.25;
    a2[i] = i*0.33;
  }

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j)
      A[i][j] = a1[i]*a2[j];
    A[i][i] += i+1;
    b[i] = -i-1;
  }

  /*------------------------------------------------------------------------*/
  /* Taping the computation of the determinant */
  trace_on(p.tag);
  /* marking indeps */
  for(int i=0; i<size; i++)
    for(int j=0; j<size; j++)
      AA[i][j] <<= (p.args[i*size+j] = A[i][j]);
  for(int i=0; i<size; i++)
    Abx[i] <<= (p.args[size*size+i] = b[i]);
  /* LU-factorization and computation of solution */
  LUfact(size,AA);
  LUsolve(size,AA,Abx);
  /* marking deps */
  for (int i = 0; i< size; ++i)
    Abx[i] >>= p.x[i];
  trace_off();
  std::cout << "  x[0] (original) : " << std::scientific << p.x[0] << "\n";

  /*------------------------------------------------------------------------*/
  /* Recomputation  */
  function(p.tag, p.depen, p.indep, p.args, p.x);
  std::cout << "  x[0] (from tape): " << std::scientific << p.x[0] << "\n";

  myfree1(b);
  myfree1(a1);
  myfree1(a2);
  myfree2(A);
  delete[] Abx;
  delete[] AA;

  return ret;
}

int apply_drivers(Problem& p) {
  int ret = 0;

  if (0 >= p.size)
    return 1;

  double** J = myalloc2(p.depen, p.indep);
  get_Jacobian(p, J);

  /*------------------------------------------------------------------------*/
  /* Z = J*U with U(i,j) = i+j */
  uint l = p.depen;
  double** U = myalloc2(p.indep, l);
  std::cout << "  --------------------------------------\n";
  std::cout << "  | Evaluate Z = J*U with U(i,j) = i+j |\n";
  std::cout << "  --------------------------------------\n";
  mat_fill_iplusj(p.indep, l, U);
  if (6 > p.size)
    mat_print("U1", p.indep, l, U);
  compute_JU(p.depen, p.indep, l, p, U, J);
  myfree2(U);

  /*------------------------------------------------------------------------*/
  /* Z = J*U with random values in U */
  std::cout << "  -----------------------------------------------\n";
  std::cout << "  | Evaluate Z = J*U with U(i,j) = rand() % 100 |\n";
  std::cout << "  -----------------------------------------------\n";
  U = myalloc2(p.indep, l);
  mat_fill_rand(p.indep, l, U);
  if (6 > p.size)
    mat_print("U", p.indep, l, U);
  compute_JU(p.depen, p.indep, l, p, U, J);
  myfree2(U);


  /*------------------------------------------------------------------------*/
  /* Z = U*J with U(i,j) = i+j */
  std::cout << "  --------------------------------------\n";
  std::cout << "  | Evaluate Z = U*J with U(i,j) = i+j |\n";
  std::cout << "  --------------------------------------\n";
  uint q = p.depen+1;
  U = myalloc2(q, l);
  mat_fill_iplusj(q, l, U);
  if (6 > p.size)
    mat_print("U", q, l, U);
  compute_UJ(q, p.depen, p.indep, p, U, J);
  myfree2(U);

  /*------------------------------------------------------------------------*/
  /* Z = U*J with random values in U */
//  std::cout << "  -----------------------------------------------\n";
//  std::cout << "  | Evaluate Z = U*J with U(i,j) = rand() % 100 |\n";
//  std::cout << "  -----------------------------------------------\n";
//  U = myalloc2(l, p.depen);
//  mat_fill_rand(l, p.depen, U);
//  if (6 > p.size)
//    mat_print("U", l, p.indep, U);
//  compute_UJ(l, p.depen, p.indep, p, U, J);
//  myfree2(U);

  /*------------------------------------------------------------------------*/
  /* Tape statistics */
  ulong tape_stats[STAT_SIZE];
  tapestats(p.tag,tape_stats);

  std::cout << "  Tape Statistics:\n";
  std::cout << "    independents            " << tape_stats[NUM_INDEPENDENTS]
            << "\n    dependents              " << tape_stats[NUM_DEPENDENTS]
            << "\n    operations              " << tape_stats[NUM_OPERATIONS]
            << "\n    operations buffer size  " << tape_stats[OP_BUFFER_SIZE]
            << "\n    locations buffer size   " << tape_stats[LOC_BUFFER_SIZE]
            << "\n    constants buffer size   " << tape_stats[VAL_BUFFER_SIZE]
            << "\n    maxlive                 " << tape_stats[NUM_MAX_LIVES]
            << "\n    valstack size           " << tape_stats[TAY_STACK_SIZE] << "\n\n";

  myfree2(J);

  /*------------------------------------------------------------------------*/
  /* That's it */
  return ret;
}

/******************************************************************************/
void mat_fill_iplusj(const uint m, const uint n, double* const* M)
{
  for (uint i = 0; i < m; ++i)
    for (uint j = 0; j < n; ++j)
      M[i][j] = i+j;
}

/******************************************************************************/
void mat_fill_rand(const uint m, const uint n, double* const* M)
{
  /* initialize random seed: */
  //srand (time(NULL));
  for (uint i = 0; i < m; ++i)
    for (uint j = 0; j < n; ++j)
      M[i][j] = rand() % 100;
}

/******************************************************************************/
void mat_print(const std::string& name, const uint m, const uint n,
               constMat M)
{
  //std::cout.precision(4);
  std::cout << "\n  Print matrix " << name << " (" << m << "x" << n << "):\n";
  for(uint i = 0; i < m ; ++i) {
    std::cout << "  " << i << ": ";
    for(uint j = 0; j < n ; ++j)
      std::cout << std::setprecision(4) << std::fixed << M[i][j] << "  ";
    std::cout << "\n";
  }
  std::cout << "\n";
}

/******************************************************************************/
int compare_mats(const uint m, const uint n, constMat jac1,
                 const std::string& name1,
                 constMat jac2, const std::string& name2)
{
  double eps = 1.E-10;
  double f;
  int ret = 0;

  std::cout << "\n  Compare results:\n";
  for (uint i = 0; i < m ; ++i) {
    for (uint j = 0; j < n ; ++j) {
      f = fabs(jac1[i][j] - jac2[i][j]);
      if (f > eps) {
        std::cout << "\tUnexpected value: expected[" << i << "][" << j << "] = "
            << jac1[i][j]
            << " vs result[" << i << "][" << j << "] = " << jac2[i][j] << "\n";
        ret = 1;
      }
    }
  }
  if (!ret)
    std::cout << "    " << name1 << " and " << name2
              << " are identical within eps " << std::scientific << eps << ".\n";
  std::cout << "\n";

  return ret;
}

/******************************************************************************/
int get_Jacobian(Problem& p, double** J)
{
  /*------------------------------------------------------------------------*/
  /* Computation of Jacobian */
  jacobian(p.tag, p.depen, p.indep, p.args, J);
  if (6 > p.size)
    mat_print("Jacobian", p.depen, p.indep, J);

  /*------------------------------------------------------------------------*/
  /* Parallel computation of Jacobian */
  double** parJ = myalloc2(p.depen, p.indep);
  par_jacobian(p.tag, p.depen, p.indep, p.args, parJ);
  if (6 > p.size)
    mat_print("Par Jacobian", p.depen, p.indep, parJ);

  /*------------------------------------------------------------------------*/
  /* Compare Jacobian and Parallel Jacobian*/
  int ret = compare_mats(p.depen, p.indep, J, "jac", parJ, "parJac");

  myfree2(parJ);
  if (ret) {
    std::cout << "Result from drivers jacobian and par_jacobian differ."
                 " Fix this, than come back. Exit.\n";
    exit(1);
  }

  return ret;
}

/******************************************************************************/
int compute_JU(uint m, uint n, uint l, Problem& p, double** U, double** J)
{
  double** Z = myalloc2(p.depen, l);
  par_jac_mat(p.tag, p.depen, p.indep, l, p.args, U, Z);
  if (6 > p.size)
    mat_print("Z=JU", p.depen, l, Z);

  /* Control result */
  double** Ztilde = myalloc2(p.depen, l);
  mat_mul(J, U, Ztilde, p.depen, p.indep, l);
  if (6 > p.size)
    mat_print("Ztilde", p.depen, l, Ztilde);
  int ret = compare_mats(p.depen, l, Z, "par_jac_mat", Ztilde, "Control Result");

  myfree2(Z);
  myfree2(Ztilde);
  return ret;
}

/******************************************************************************/
/* U \in R^{m, n}
 * J \in R^{n, l}
 * Z \in R^{m, l}
 */
int compute_UJ(uint m, uint n, uint l, Problem& p, double** U, double** J)

{
  int ret = 0;
  double** Z = myalloc2(m, l);
  par_mat_jac(p.tag, p.depen, p.indep, m, p.args, U, Z);
  if (6 > p.size)
    mat_print("Z=UJ", m, l, Z);

  /* Control result */
  double** Ztilde = myalloc2(m, l);
  mat_mul(U, J, Ztilde, m, n, l);
  if (6 > p.size)
    mat_print("Ztilde", m, l, Ztilde);
  ret = compare_mats(m, l, Z, "par_mat_jac", Ztilde, "Control Result");

  myfree2(Z);
  myfree2(Ztilde);
  return ret;
}

/**
 * Computes C = AB whereas A \in R^{m,n}, B \in R^{n,l} and C \in R^{m,l}.
 */
void mat_mul(constMat A, constMat B, double* const* C, const uint m,
             const uint n, const uint l)
{
  // Make sure C = zeros(m, l)
  for (uint i = 0; i < m; ++i) {
    for (uint j = 0; j < l; ++j) {
      C[i][j] = 0.0;
    }
  }

  for (uint i = 0; i < m; ++i) {
    for (uint k = 0; k < l; ++k) {
      for (uint j = 0; j < n; ++j) {
        C[i][k] += A[i][j]*B[j][k];
      }
    }
  }
}
