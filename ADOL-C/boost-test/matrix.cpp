/*
   File to explicitely test matrixView from adalloc.cpp
*/

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

namespace tt = boost::test_tools;

#include "const.h"
#include <adolc/adolc.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
struct ThrowingMoveElement {
  ThrowingMoveElement() = default;
  ThrowingMoveElement(ThrowingMoveElement &&) {
    throw std::runtime_error("element move failed");
  }
  ThrowingMoveElement &operator=(ThrowingMoveElement &&) noexcept = default;
};
} // namespace

BOOST_AUTO_TEST_SUITE(adalloc_cont)

/**************************** Vector Container ******************************/

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_2_1_double) {
  std::vector<double> container{1.0, 2.0};

  /* View to check:
    ( ( 1.0 ) )
    ( ( 2.0 ) )
  */
  auto containerView = matrixView(container, 2, 1);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_1_2_double) {
  std::vector<double> container{1.0, 2.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) )
  */
  auto containerView = matrixView(container, 1, 2);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_2_2_double) {
  std::vector<double> container{1.0, 2.0, 3.0, 4.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) )
    ( ( 3.0 ) ( 4.0 ) )
  */
  auto containerView = matrixView(container, 2, 2);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_1_3_double) {
  std::vector<double> container{1.0, 2.0, 3.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) ( 3.0 ) )
  */
  auto containerView = matrixView(container, 1, 3);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_3_1_double) {
  std::vector<double> container{1.0, 2.0, 3.0};

  /* View to check:
    ( ( 1.0 ) )
    ( ( 2.0 ) )
    ( ( 3.0 ) )
  */
  auto containerView = matrixView(container, 3, 1);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[2][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_3_2_double) {
  std::vector<double> container{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) )
    ( ( 3.0 ) ( 4.0 ) )
    ( ( 5.0 ) ( 6.0 ) )
  */
  auto containerView = matrixView(container, 3, 2);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[2][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_2_3_double) {
  std::vector<double> container{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) ( 3.0 ) )
    ( ( 4.0 ) ( 5.0 ) ( 6.0 ) )
  */
  auto containerView = matrixView(container, 2, 3);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_3_3_double) {
  std::vector<double> container{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) ( 3.0 ) )
    ( ( 4.0 ) ( 5.0 ) ( 6.0 ) )
    ( ( 7.0 ) ( 8.0 ) ( 9.0 ) )
  */
  auto containerView = matrixView(container, 3, 3);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
  BOOST_TEST(container[6] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[7] == containerView[2][1], tt::tolerance(tol));
  BOOST_TEST(container[8] == containerView[2][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_3_2_int) {
  std::vector<double> container{1, 2, 3, 4, 5, 6};

  /* View to check:
    ( ( 1 ) ( 2 ) )
    ( ( 3 ) ( 4 ) )
    ( ( 5 ) ( 6 ) )
  */
  auto containerView = matrixView(container, 3, 2);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[2][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_2_3_int) {
  std::vector<int> container{1, 2, 3, 4, 5, 6};

  /* View to check:
    ( ( 1 ) ( 2 ) ( 3 ) )
    ( ( 4 ) ( 5 ) ( 6 ) )
  */
  auto containerView = matrixView(container, 2, 3);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Vector_3_3_int) {
  std::vector<int> container{1, 2, 3, 4, 5, 6, 7, 8, 9};

  /* View to check:
    ( ( 1 ) ( 2 ) ( 3 ) )
    ( ( 4 ) ( 5 ) ( 6 ) )
    ( ( 7 ) ( 8 ) ( 9 ) )
  */
  auto containerView = matrixView(container, 3, 3);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
  BOOST_TEST(container[6] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[7] == containerView[2][1], tt::tolerance(tol));
  BOOST_TEST(container[8] == containerView[2][2], tt::tolerance(tol));
}

/****************************************************************************/

/***************************** Array Container ******************************/

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_2_1_double) {
  std::array<double, 2> container{1.0, 2.0};

  /* View to check:
    ( ( 1.0 ) )
    ( ( 2.0 ) )
  */
  auto containerView = matrixView<2, 1>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_1_2_double) {
  std::array<double, 2> container{1.0, 2.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) )
  */
  auto containerView = matrixView<1, 2>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_2_2_double) {
  std::array<double, 4> container{1.0, 2.0, 3.0, 4.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) )
    ( ( 3.0 ) ( 4.0 ) )
  */
  auto containerView = matrixView<2, 2>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_1_3_double) {
  std::array<double, 3> container{1.0, 2.0, 3.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) ( 3.0 ) )
  */
  auto containerView = matrixView<1, 3>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_3_1_double) {
  std::array<double, 3> container{1.0, 2.0, 3.0};

  /* View to check:
    ( ( 1.0 ) )
    ( ( 2.0 ) )
    ( ( 3.0 ) )
  */
  auto containerView = matrixView<3, 1>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[2][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_3_2_double) {
  std::array<double, 6> container{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) )
    ( ( 3.0 ) ( 4.0 ) )
    ( ( 5.0 ) ( 6.0 ) )
  */
  auto containerView = matrixView<3, 2>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[2][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_2_3_double) {
  std::array<double, 6> container{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) ( 3.0 ) )
    ( ( 4.0 ) ( 5.0 ) ( 6.0 ) )
  */
  auto containerView = matrixView<2, 3>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_3_3_double) {
  std::array<double, 9> container{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  /* View to check:
    ( ( 1.0 ) ( 2.0 ) ( 3.0 ) )
    ( ( 4.0 ) ( 5.0 ) ( 6.0 ) )
    ( ( 7.0 ) ( 8.0 ) ( 9.0 ) )
  */
  auto containerView = matrixView<3, 3>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
  BOOST_TEST(container[6] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[7] == containerView[2][1], tt::tolerance(tol));
  BOOST_TEST(container[8] == containerView[2][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_3_2_int) {
  std::array<int, 6> container{1, 2, 3, 4, 5, 6};

  /* View to check:
    ( ( 1 ) ( 2 ) )
    ( ( 3 ) ( 4 ) )
    ( ( 5 ) ( 6 ) )
  */
  auto containerView = matrixView<3, 2>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[2][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_2_3_int) {
  std::array<int, 6> container{1, 2, 3, 4, 5, 6};

  /* View to check:
    ( ( 1 ) ( 2 ) ( 3 ) )
    ( ( 4 ) ( 5 ) ( 6 ) )
  */
  auto containerView = matrixView<2, 3>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_View_of_Array_3_3_int) {
  std::array<int, 9> container{1, 2, 3, 4, 5, 6, 7, 8, 9};

  /* View to check:
    ( ( 1 ) ( 2 ) ( 3 ) )
    ( ( 4 ) ( 5 ) ( 6 ) )
    ( ( 7 ) ( 8 ) ( 9 ) )
  */
  auto containerView = matrixView<3, 3>(container);

  BOOST_TEST(container[0] == containerView[0][0], tt::tolerance(tol));
  BOOST_TEST(container[1] == containerView[0][1], tt::tolerance(tol));
  BOOST_TEST(container[2] == containerView[0][2], tt::tolerance(tol));
  BOOST_TEST(container[3] == containerView[1][0], tt::tolerance(tol));
  BOOST_TEST(container[4] == containerView[1][1], tt::tolerance(tol));
  BOOST_TEST(container[5] == containerView[1][2], tt::tolerance(tol));
  BOOST_TEST(container[6] == containerView[2][0], tt::tolerance(tol));
  BOOST_TEST(container[7] == containerView[2][1], tt::tolerance(tol));
  BOOST_TEST(container[8] == containerView[2][2], tt::tolerance(tol));
}

/****************************************************************************/

/****************************** Matrix(Vector) ******************************/

BOOST_AUTO_TEST_CASE(Matrix_2_1_double) {
  auto test = Matrix<double>(2, 1);

  BOOST_TEST(0.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == test[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_1_2_double) {
  auto test = Matrix<double>(1, 2);

  BOOST_TEST(0.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == test[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_2_1_double_fill) {
  auto test = Matrix<double>(2, 1);
  test.fill(3.0);

  BOOST_TEST(3.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_1_2_double_fill) {
  auto test = Matrix<double>(1, 2);
  test.fill(3.0);

  BOOST_TEST(3.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_1_2_double_Copy_fill) {
  auto test = Matrix<double>(1, 2);
  test.fill(2.0);
  auto copy = Matrix(test);

  BOOST_TEST(2.0 == copy[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[0][1], tt::tolerance(tol));
  copy[0][0] = 9.0;
  BOOST_TEST(test[0][0] == 2.0);
}

BOOST_AUTO_TEST_CASE(Matrix_2_double_Copy_ass) {
  auto test = Matrix<double>(2, 2, 2.0);
  Matrix<double> copy{2, 2};
  copy = test;

  BOOST_TEST(2.0 == copy[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[0][1], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[1][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[1][1], tt::tolerance(tol));
  copy[0][0] = 9.0;
  BOOST_TEST(test[0][0] == 2.0);
}

BOOST_AUTO_TEST_CASE(Matrix_2_double_Move) {
  auto test = Matrix<double>(2, 2, 2.0);
  auto moved(std::move(test));

  BOOST_TEST(2.0 == moved[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[0][1], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_2_double_Move_ass) {
  auto test = Matrix<double>(2, 2, 2.0);
  Matrix<double> moved;
  moved = std::move(test);

  BOOST_TEST(2.0 == moved[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[0][1], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_2_2_double_man_fill) {
  auto test = Matrix<double>(2);
  test[0][0] = 1.0;
  test[0][1] = 2.0;
  test[1][0] = 3.0;
  test[1][1] = 4.0;

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(4.0 == test[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_3_3_double_man_fill) {
  auto test = Matrix<double>(3);
  test[0][0] = 1.0;
  test[0][1] = 2.0;
  test[0][2] = 3.0;
  test[1][0] = 4.0;
  test[1][1] = 5.0;
  test[1][2] = 6.0;
  test[2][0] = 7.0;
  test[2][1] = 8.0;
  test[2][2] = 9.0;

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[0][2], tt::tolerance(tol));
  BOOST_TEST(4.0 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][2], tt::tolerance(tol));
  BOOST_TEST(7.0 == test[2][0], tt::tolerance(tol));
  BOOST_TEST(8.0 == test[2][1], tt::tolerance(tol));
  BOOST_TEST(9.0 == test[2][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_2_2_int_man_fill) {
  auto test = Matrix<int>(2);
  test[0][0] = 1;
  test[0][1] = 2;
  test[1][0] = 3;
  test[1][1] = 4;

  BOOST_TEST(1 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(4 == test[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_3_3_int_man_fill) {
  auto test = Matrix<int>(3);
  test[0][0] = 1;
  test[0][1] = 2;
  test[0][2] = 3;
  test[1][0] = 4;
  test[1][1] = 5;
  test[1][2] = 6;
  test[2][0] = 7;
  test[2][1] = 8;
  test[2][2] = 9;

  BOOST_TEST(1 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3 == test[0][2], tt::tolerance(tol));
  BOOST_TEST(4 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(5 == test[1][1], tt::tolerance(tol));
  BOOST_TEST(6 == test[1][2], tt::tolerance(tol));
  BOOST_TEST(7 == test[2][0], tt::tolerance(tol));
  BOOST_TEST(8 == test[2][1], tt::tolerance(tol));
  BOOST_TEST(9 == test[2][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Matrix_from_Existing) {
  std::vector<double> data{1.0, 2.0, 3.0, 4.0};
  std::vector<double *> view(2);
  for (size_t i = 0; i < 2; i++) {
    view[i] = data.data() + i * 2;
  }
  auto test = Matrix<double>(std::move(view), std::move(data));

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(4.0 == test[1][1], tt::tolerance(tol));
  BOOST_TEST(test.shape().first * test.shape().second == 4, tt::tolerance(tol));
}

/****************************************************************************/

/****************************** Array Matrix ********************************/

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_1_double) {
  auto test = Matrix<double, 2, 1>{};

  BOOST_TEST(0.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == test[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_1_2_double) {
  auto test = Matrix<double, 1, 2>{};

  BOOST_TEST(0.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == test[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_1_double_fill) {
  auto test = Matrix<double, 2, 1>(1.0);

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == test[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_1_2_double_fill) {
  auto test = Matrix<double, 1, 2>(1.0);

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == test[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_2_double_man_fill) {
  auto test = Matrix<double, 2>{};
  test[0][0] = 1.0;
  test[0][1] = 2.0;
  test[1][0] = 3.0;
  test[1][1] = 4.0;

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(4.0 == test[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_3_3_double_man_fill) {
  auto test = Matrix<double, 3>{};
  test[0][0] = 1.0;
  test[0][1] = 2.0;
  test[0][2] = 3.0;
  test[1][0] = 4.0;
  test[1][1] = 5.0;
  test[1][2] = 6.0;
  test[2][0] = 7.0;
  test[2][1] = 8.0;
  test[2][2] = 9.0;

  BOOST_TEST(1.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[0][2], tt::tolerance(tol));
  BOOST_TEST(4.0 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][2], tt::tolerance(tol));
  BOOST_TEST(7.0 == test[2][0], tt::tolerance(tol));
  BOOST_TEST(8.0 == test[2][1], tt::tolerance(tol));
  BOOST_TEST(9.0 == test[2][2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_1_double_Copy) {
  auto test = Matrix<double, 2, 1>(2.0);
  auto copy = Matrix(test);

  BOOST_TEST(2.0 == copy[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_1_2_double_Copy_fill) {
  auto test = Matrix<double, 1, 2>(2.0);
  auto copy = Matrix(test);
  copy.fill(4.0);

  BOOST_TEST(2.0 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(4.0 == copy[0][0], tt::tolerance(tol));
  BOOST_TEST(4.0 == copy[0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_double_Copy_ass) {
  auto test = Matrix<double, 2, 2>(2.0);
  Matrix<double, 2, 2> copy;
  copy = test;

  BOOST_TEST(2.0 == copy[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[0][1], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[1][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == copy[1][1], tt::tolerance(tol));
  copy[0][0] = 9.0;
  BOOST_TEST(test[0][0] == 2.0);
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_double_Move) {
  auto test = Matrix<double, 2, 2>(2.0);
  auto moved(std::move(test));

  BOOST_TEST(2.0 == moved[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[0][1], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][1], tt::tolerance(tol));
  moved[0][0] = 9.0;
  BOOST_TEST(test[0][0] == 2.0);
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_double_Move_ass) {
  auto test = Matrix<double, 2, 2>(2.0);
  Matrix<double, 2, 2> moved;
  moved = std::move(test);

  BOOST_TEST(2.0 == moved[0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[0][1], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == moved[1][1], tt::tolerance(tol));
  moved[0][0] = 9.0;
  BOOST_TEST(test[0][0] == 2.0);
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_2_2_int_man_fill) {
  auto test = Matrix<int, 2, 2>{};
  test[0][0] = 1;
  test[0][1] = 2;
  test[1][0] = 3;
  test[1][1] = 4;

  BOOST_TEST(1 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(4 == test[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_3_3_int_man_fill) {
  auto test = Matrix<int, 3, 3>{};
  test[0][0] = 1;
  test[0][1] = 2;
  test[0][2] = 3;
  test[1][0] = 4;
  test[1][1] = 5;
  test[1][2] = 6;
  test[2][0] = 7;
  test[2][1] = 8;
  test[2][2] = 9;

  BOOST_TEST(1 == test[0][0], tt::tolerance(tol));
  BOOST_TEST(2 == test[0][1], tt::tolerance(tol));
  BOOST_TEST(3 == test[0][2], tt::tolerance(tol));
  BOOST_TEST(4 == test[1][0], tt::tolerance(tol));
  BOOST_TEST(5 == test[1][1], tt::tolerance(tol));
  BOOST_TEST(6 == test[1][2], tt::tolerance(tol));
  BOOST_TEST(7 == test[2][0], tt::tolerance(tol));
  BOOST_TEST(8 == test[2][1], tt::tolerance(tol));
  BOOST_TEST(9 == test[2][2], tt::tolerance(tol));
}

/****************************************************************************/

/******************************* Unit Vector ********************************/

// Tests for std::array unit vector
BOOST_AUTO_TEST_CASE(Array_Unit_Vector_1_0) {
  auto unit = unitVector<double, 2>(1);

  BOOST_TEST(1.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Array_Unit_Vector_0_1) {
  auto unit = unitVector<double, 2>(2);

  BOOST_TEST(0.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Array_Unit_Vector_1_0_int) {
  auto unit = unitVector<int, 2>(1);

  BOOST_TEST(1 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Array_Unit_Vector_0_1_int) {
  auto unit = unitVector<int, 2>(2);

  BOOST_TEST(0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(1 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Array_Unit_Vector_1_0_0) {
  auto unit = unitVector<double, 3>(1);

  BOOST_TEST(1.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Array_Unit_Vector_0_1_0) {
  auto unit = unitVector<double, 3>(2);

  BOOST_TEST(0.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Array_Unit_Vector_0_0_1) {
  auto unit = unitVector<double, 3>(3);

  BOOST_TEST(0.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[2], tt::tolerance(tol));
}

// Tests for std::vector unit vector
BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_1_0) {
  auto unit = unitVector(2, 1);

  BOOST_TEST(1.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_0_1) {
  auto unit = unitVector(2, 2);

  BOOST_TEST(0.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_1_0_int) {
  auto unit = unitVector<int>(2, 1);

  BOOST_TEST(1 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_0_1_int) {
  auto unit = unitVector<int>(2, 2);

  BOOST_TEST(0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(1 == unit[1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_1_0_0) {
  auto unit = unitVector(3, 1);

  BOOST_TEST(1.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_0_1_0) {
  auto unit = unitVector(3, 2);

  BOOST_TEST(0.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(VectorLikeContainer_Unit_Vector_0_0_1) {
  auto unit = unitVector(3, 3);

  BOOST_TEST(0.0 == unit[0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[2], tt::tolerance(tol));
}

/****************************************************************************/

/***************************** Identity Matrix ******************************/

// std::vector version
BOOST_AUTO_TEST_CASE(Unit_Matrix_2) {
  auto unit = unitMatrix(2);

  BOOST_TEST(1.0 == unit[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[0][1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Unit_Matrix_2_int) {
  auto unit = unitMatrix<int>(2);

  BOOST_TEST(1 == unit[0][0], tt::tolerance(tol));
  BOOST_TEST(0 == unit[0][1], tt::tolerance(tol));
  BOOST_TEST(0 == unit[1][0], tt::tolerance(tol));
  BOOST_TEST(1 == unit[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Unit_Matrix_3) {
  auto unit = unitMatrix(3);

  BOOST_TEST(1.0 == unit[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[0][1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[0][2], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1][1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1][2], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2][1], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[2][2], tt::tolerance(tol));
}

// std::array version
BOOST_AUTO_TEST_CASE(Unit_ArrayMatrix_2) {
  auto unit = unitMatrix<double, 2>();

  BOOST_TEST(1.0 == unit[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[0][1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Unit_ArrayMatrix_2_int) {
  auto unit = unitMatrix<int, 2>();

  BOOST_TEST(1 == unit[0][0], tt::tolerance(tol));
  BOOST_TEST(0 == unit[0][1], tt::tolerance(tol));
  BOOST_TEST(0 == unit[1][0], tt::tolerance(tol));
  BOOST_TEST(1 == unit[1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Unit_ArrayMatrix_3) {
  auto unit = unitMatrix<double, 3>();

  BOOST_TEST(1.0 == unit[0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[0][1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[0][2], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[1][1], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[1][2], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == unit[2][1], tt::tolerance(tol));
  BOOST_TEST(1.0 == unit[2][2], tt::tolerance(tol));
}

/****************************************************************************/

/******************************** Tensor *********************************/

BOOST_AUTO_TEST_CASE(Tensor_2_1_1_double) {
  auto test = Tensor(2, 1, 1);

  BOOST_TEST(0.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(0.0 == test[1][0][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_1_2_1_double_man_fill) {
  auto test = Tensor(1, 2, 1);
  test.fill(1.0);

  BOOST_TEST(1.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(1.0 == test[0][1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_1_1_2_double_fill) {
  auto test = Tensor(1, 1, 2, 2.0);

  BOOST_TEST(2.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_1_double_fill) {
  auto test = Tensor(2, 2, 1);
  test.fill(5.0);

  BOOST_TEST(5.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[0][1][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][0][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][1][0], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_1_2_double_man_fill) {
  auto test = Tensor(2, 1, 2);
  test.fill(5.0);

  BOOST_TEST(5.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[0][0][1], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][0][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][0][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_1_2_2_double_man_fill) {
  auto test = Tensor(1, 2, 2);
  test.fill(5.0);

  BOOST_TEST(5.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[0][0][1], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[0][1][0], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[0][1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_double_man_fill) {
  auto test = Tensor(2, 2, 2);
  test.fill(6.0);

  BOOST_TEST(6.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[0][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[0][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[0][1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_double_man_copy_fill) {
  auto test = Tensor(2, 2, 2);
  test.fill(6.0);
  auto copy = Tensor(test);

  BOOST_TEST(6.0 == copy[0][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[0][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[0][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[0][1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][1][1], tt::tolerance(tol));
  copy[0][0][0] = 9.0;
  BOOST_TEST(test[0][0][0] == 6.0);
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_double_man_copy_ass) {
  auto test = Tensor<double>(2, 2, 2, 6.0);
  Tensor<double> copy{2, 2, 2};
  copy = test;

  BOOST_TEST(6.0 == copy[0][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[0][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[0][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[0][1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == copy[1][1][1], tt::tolerance(tol));
  copy[0][0][0] = 9.0;
  BOOST_TEST(test[0][0][0] == 6.0);
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_double_man_Move) {
  auto test = Tensor<double>(2, 2, 2, 6.0);
  auto moved(std::move(test));

  BOOST_TEST(6.0 == moved[0][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[0][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[0][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[0][1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_double_man_Move_ass) {
  auto test = Tensor<double>{2, 2, 2, 6.0};
  Tensor<double> moved;
  moved = std::move(test);

  BOOST_TEST(6.0 == moved[0][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[0][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[0][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[0][1][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][0][1], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][1][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == moved[1][1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_man_fill) {
  auto test = Tensor(2, 2, 2, 1.0);
  // Give Values
  test[0][0][1] = 2.0;
  test[0][1][0] = 3.0;
  test[0][1][1] = 4.0;
  test[1][0][0] = 5.0;
  test[1][0][1] = 6.0;
  test[1][1][0] = 7.0;
  test[1][1][1] = 8.0;

  // Check Correctness
  BOOST_TEST(1.0 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(2.0 == test[0][0][1], tt::tolerance(tol));
  BOOST_TEST(3.0 == test[0][1][0], tt::tolerance(tol));
  BOOST_TEST(4.0 == test[0][1][1], tt::tolerance(tol));
  BOOST_TEST(5.0 == test[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6.0 == test[1][0][1], tt::tolerance(tol));
  BOOST_TEST(7.0 == test[1][1][0], tt::tolerance(tol));
  BOOST_TEST(8.0 == test[1][1][1], tt::tolerance(tol));
}

BOOST_AUTO_TEST_CASE(Tensor_2_2_2_man_fill_int) {
  auto test = Tensor(2, 2, 2, 1);
  // Give Values
  test[0][0][1] = 2;
  test[0][1][0] = 3;
  test[0][1][1] = 4;
  test[1][0][0] = 5;
  test[1][0][1] = 6;
  test[1][1][0] = 7;
  test[1][1][1] = 8;

  // Check Correctness
  BOOST_TEST(1 == test[0][0][0], tt::tolerance(tol));
  BOOST_TEST(2 == test[0][0][1], tt::tolerance(tol));
  BOOST_TEST(3 == test[0][1][0], tt::tolerance(tol));
  BOOST_TEST(4 == test[0][1][1], tt::tolerance(tol));
  BOOST_TEST(5 == test[1][0][0], tt::tolerance(tol));
  BOOST_TEST(6 == test[1][0][1], tt::tolerance(tol));
  BOOST_TEST(7 == test[1][1][0], tt::tolerance(tol));
  BOOST_TEST(8 == test[1][1][1], tt::tolerance(tol));
}

/****************************************************************************/

BOOST_AUTO_TEST_CASE(Matrix_Empty_And_Moved_From_States) {
  Matrix<double> empty;
  Matrix<double> copy(empty);
  BOOST_TEST(copy.shape().first == 0u);
  BOOST_TEST(copy.shape().second == 0u);

  Matrix<double> source(2, 3, 7.0);
  Matrix<double> moved(std::move(source));
  BOOST_TEST(source.shape().first == 0u);
  BOOST_TEST(source.shape().second == 0u);
  std::span<double *> sourceRows = source;
  BOOST_TEST(sourceRows.empty());

  // Moving an already moved-from object must also produce an empty matrix.
  copy = Matrix<double>(1, 1);
  copy = std::move(source);
  BOOST_TEST(copy.shape().first == 0u);
  BOOST_TEST(copy.shape().second == 0u);

  source = std::move(moved);
  auto *self = &source;
  source = std::move(*self);
  const auto &view = source;
  BOOST_TEST(view.shape().first == 2u);
  BOOST_TEST(view.shape().second == 3u);
  BOOST_TEST(view[1][2] == 7.0);
  BOOST_TEST(view.data()[1] == view.data()[0] + 3);
}

BOOST_AUTO_TEST_CASE(Tensor_Empty_And_Moved_From_States) {
  Tensor<double> empty;
  Tensor<double> copy(empty);
  for (auto extent : copy.shape())
    BOOST_TEST(extent == 0u);

  Tensor<double> source(2, 3, 4);
  source[1][2][3] = 11.0;
  Tensor<double> moved(std::move(source));
  for (auto extent : source.shape())
    BOOST_TEST(extent == 0u);
  std::span<double **> sourceRows = source;
  BOOST_TEST(sourceRows.empty());

  copy = Tensor<double>(1, 1, 1);
  copy = std::move(source);
  for (auto extent : copy.shape())
    BOOST_TEST(extent == 0u);

  source = std::move(moved);
  auto *self = &source;
  source = std::move(*self);
  const auto &view = source;
  BOOST_TEST(view.shape()[0] == 2u);
  BOOST_TEST(view.shape()[1] == 3u);
  BOOST_TEST(view.shape()[2] == 4u);
  BOOST_TEST(view[1][2][3] == 11.0);
  BOOST_TEST(view.data()[1][2] == view.data()[0][0] + 20);
}

BOOST_AUTO_TEST_CASE(ArrayMatrix_Move_Propagates_Element_Exception) {
  using ThrowingMatrix = Matrix<ThrowingMoveElement, 1, 1>;
  static_assert(!std::is_nothrow_move_constructible_v<ThrowingMatrix>);
  static_assert(std::is_nothrow_move_assignable_v<ThrowingMatrix>);
  ThrowingMatrix source;
  BOOST_CHECK_THROW(ThrowingMatrix(std::move(source)), std::runtime_error);
}

BOOST_AUTO_TEST_SUITE_END()
