#include <adolc/drivers/absnormalform.h>

namespace ADOLC {

DenseShape getShapeFromTape(short tapeId) {
  const ValueTape &tape = findTape(tapeId);
  return DenseShape{tape.tapestats(TapeInfos::NUM_DEPENDENTS),
                    tape.tapestats(TapeInfos::NUM_INDEPENDENTS),
                    tape.tapestats(TapeInfos::NUM_SWITCHES)};
}
void AbsNormalForm::clear() {
  shape = {};

  Y.clear();
  J.clear();
  Z.clear();
  L.clear();

  Y_storage.clear();
  J_storage.clear();
  Z_storage.clear();
  L_storage.clear();

  y.clear();
  z.clear();
  cy.clear();
  cz.clear();
}

void AbsNormalForm::resize(AbsNormalForm::Shape dims) {
  shape = dims;

  Y_storage.resize(dims.m * dims.n, 0.0);
  J_storage.resize(dims.m * dims.s, 0.0);
  Z_storage.resize(dims.s * dims.n, 0.0);
  L_storage.resize(dims.s * dims.s, 0.0);

  Y.resize(dims.m);
  J.resize(dims.m);
  Z.resize(dims.s);
  L.resize(dims.s);

  for (size_t i = 0; i < dims.m; ++i) {
    Y[i] = dims.n == 0 ? nullptr : Y_storage.data() + (i * dims.n);
  }
  for (size_t i = 0; i < dims.m; ++i) {
    J[i] = dims.s == 0 ? nullptr : J_storage.data() + (i * dims.s);
  }
  for (size_t i = 0; i < dims.s; ++i) {
    Z[i] = dims.n == 0 ? nullptr : Z_storage.data() + (i * dims.n);
  }
  for (size_t i = 0; i < dims.s; ++i) {
    L[i] = dims.s == 0 ? nullptr : L_storage.data() + (i * dims.s);
  }

  y.resize(dims.m, 0.0);
  z.resize(dims.s, 0.0);
  cy.resize(dims.m, 0.0);
  cz.resize(dims.s, 0.0);
}

} // namespace ADOLC
