/*
This file defines an abs-smooth example function f and computes its jacobian
pattern

dimIn = 2, dimOut = 1
Input x = (x0, x1)

Let z = |x1|. Then

    f8(x) = max( -100,  3x0 + 2z,  2x0 + 5z ).

sparsity pattern (crs):
[3, 0, 2, 3, 4]
[1, 1]
[2, 0, 2]
[3, 0, 2, 3]
*/

#include <adolc/adolc.h>
#include <array>
#include <numeric>

struct UnAllocated {};
struct Allocated {};

template <typename State = UnAllocated> struct ANFProblem;
template <> struct ANFProblem<Allocated>;
template <> struct ANFProblem<UnAllocated> {
  static constexpr size_t dimIn = 2;
  static constexpr std::array<double, dimIn> in = {2.0, 2.0};
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 4> crs = {
      {{4, 0, 2, 3, 4}, {1, 1, 0, 0, 0}, {2, 0, 2, 0, 0}, {3, 0, 2, 3, 0}}};

  std::array<double, dimOut> out;

  constexpr ANFProblem() = default;
  ANFProblem<Allocated> allocateBuffers(short numSwitchingVars);
};

template <> struct ANFProblem<Allocated> {
  static constexpr size_t dimIn = 2;
  static constexpr size_t dimOut = 1;
  static constexpr std::array<std::array<size_t, 5>, 4> crs = {
      {{4, 0, 2, 3, 4}, {1, 1, 0, 0, 0}, {2, 0, 2, 0, 0}, {3, 0, 2, 3, 0}}};

  std::array<double, dimIn> in;
  std::array<double, dimOut> out;

  int numSwitchingVars;
  std::vector<double> z;
  std::vector<double> cz;
  std::vector<double> cy;

  double **J, **Y;
  double **Z, **L;

  std::vector<double> d;
  std::vector<double> g;
  double **gradz;

  std::vector<short> sigma_x;
  std::vector<short> sigma_g;

  ANFProblem(ANFProblem &anfProblem) = delete;
  ANFProblem(ANFProblem &&anfProblem) = delete;

  ANFProblem operator=(ANFProblem &anfProblem) = delete;
  ANFProblem operator=(ANFProblem &&anfProblem) = delete;

  ~ANFProblem() {
    myfree2(Z);
    myfree2(L);
    myfree2(J);
    myfree2(Y);
    myfree2(gradz);
  }
  ANFProblem(short numSVars, const ANFProblem<UnAllocated> &base)
      : in(base.in), out(base.out) {

    numSwitchingVars = numSVars;
    sigma_x.resize(numSwitchingVars);
    sigma_g.resize(numSwitchingVars);
    z.resize(numSwitchingVars);

    cz.resize(numSwitchingVars);
    cy.resize(dimOut);
    Z = myalloc2(numSwitchingVars, dimIn);
    L = myalloc2(numSwitchingVars, numSwitchingVars);
    J = myalloc2(dimOut, numSwitchingVars);
    Y = myalloc2(dimOut, dimIn);

    d.resize(dimIn);
    g.resize(dimIn);
    gradz = myalloc2(numSwitchingVars, dimIn);
  }
};

ANFProblem<Allocated>
ANFProblem<UnAllocated>::allocateBuffers(short numSwitchingVars) {
  return ANFProblem<Allocated>(numSwitchingVars, *this);
}
void computeANF(ANFProblem<Allocated> &anfProblemAlloc, short tapeId) {
  zos_pl_forward(tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn, 1,
                 anfProblemAlloc.in.data(), anfProblemAlloc.out.data(),
                 anfProblemAlloc.z.data());

  abs_normal(tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn,
             anfProblemAlloc.numSwitchingVars, anfProblemAlloc.in.data(),
             anfProblemAlloc.out.data(), anfProblemAlloc.z.data(),
             anfProblemAlloc.cz.data(), anfProblemAlloc.cy.data(),
             anfProblemAlloc.Y, anfProblemAlloc.J, anfProblemAlloc.Z,
             anfProblemAlloc.L);
}

void taping(ANFProblem<UnAllocated> &anfProblem) {
  std::vector<adouble> x(anfProblem.dimIn);
  std::vector<adouble> y(anfProblem.dimOut);
  x[0] <<= anfProblem.in[0];
  x[1] <<= anfProblem.in[1];
  adouble z1 = fabs(x[1]);
  y[0] = fmax(fmax(-100, 3 * x[0] + 2 * z1), 2 * x[0] + 5 * z1);
  y[0] >>= anfProblem.out[0];
}

void problem() {
  ANFProblem<UnAllocated> anfProblem{};
  const auto tapeId = createNewTape();
  currentTape().enableMinMaxUsingAbs();
  trace_on(tapeId);
  taping(anfProblem);
  trace_off();

  int numSwitchingVars = get_num_switches(tapeId);
  auto anfProblemAlloc = anfProblem.allocateBuffers(numSwitchingVars);
  computeANF(anfProblemAlloc, tapeId);

  std::vector<uint *> crs(anfProblemAlloc.dimOut +
                          anfProblemAlloc.numSwitchingVars);
  std::span<uint *> crsSpan(crs);
  ADOLC::Sparse::absnormal_jac_pat(
      tapeId, anfProblemAlloc.dimOut, anfProblemAlloc.dimIn,
      anfProblemAlloc.numSwitchingVars, anfProblemAlloc.in.data(), crsSpan);

  std::cout << "Jacobian Pattern (ANF): " << std::endl;
  for (int i = 0; i < anfProblemAlloc.dimOut + anfProblemAlloc.numSwitchingVars;
       i++) {
    std::cout << "dependent variable: " << i << ": "
              << "number of non-zeros: " << crs[i][0] << " (should be "
              << anfProblemAlloc.crs[i][0] << ")\n";
    std::cout << "non-zero indices: ";
    for (int j = 1; j <= crs[i][0]; j++)
      std::cout << crs[i][j] << ", ";

    std::cout << "   (should be ";
    for (int j = 1; j <= anfProblemAlloc.crs[i][0]; j++)
      std::cout << anfProblemAlloc.crs[i][j] << ", ";
    std::cout << ")\n";
  }
}
//******************************************************************

int main() {
  problem();
  return 0;
}
