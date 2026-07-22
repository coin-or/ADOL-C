/* Example for use of containers found in adalloc.h */

#include <adolc/adalloc.h> // For Matrix etc.
#include <adolc/adolc.h>
#include <adolc/drivers/drivers.h>
#include <adolc/interfaces.h>
#include <iostream>
#include <string_view>
#include <vector>

/****************************************************************************/
/*                                                           HELP FUNCTIONS */

/* @brief Test function
 *
 * Computes the sum of squares of the elements in the input vector.
 *
 * @param vec Input vector.
 * @return Sum of squares of the elements in the input vector.
 */
template <typename T> T square_sum(const std::vector<T> &vec) {
  T sum = 0;
  for (const auto &val : vec) {
    sum += val * val;
  }
  return sum;
}

/* @brief Print matrix View-Version
 *
 * Prints the contents of a matrix to the console.
 *
 * @param description Description of the matrix.
 * @param matrix Matrix to print.
 * @param dimy Number of rows in the matrix.
 * @param dimx Number of columns in the matrix.
 */
void printMatrix(std::string_view description, std::vector<double *> matrix,
                 size_t dimy, size_t dimx) {
  std::cout << description << " with dimensions (" << dimy << ", " << dimx
            << "):\n";
  for (int i = 0; i < dimy; ++i) {
    for (int j = 0; j < dimx; ++j) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}
/* @brief Print matrix (overload) Matrix-Version
 *
 * Prints the contents of a matrix to the console.
 *
 * @param description Description of the matrix.
 * @param matrix Matrix to print.
 */
template <typename T = double, size_t dimY = std::dynamic_extent,
          size_t dimX = dimY>
void printMatrix(std::string_view description, Matrix<T, dimY, dimX> &matrix) {
  std::cout << description << " with dimensions (" << matrix.shape().first
            << ", " << matrix.shape().second << "):\n";
  for (int i = 0; i < matrix.shape().first; ++i) {
    for (int j = 0; j < matrix.shape().second; ++j) {
      std::cout << matrix[i][j] << " ";
    }
    std::cout << "\n";
  }
}
void printVector(std::string_view description, const std::vector<double> &vec) {
  std::cout << description << "\n";
  for (const auto &val : vec) {
    std::cout << val << " ";
  }
  std::cout << "\n";
}
template <typename T = double, size_t dimY = std::dynamic_extent,
          size_t dimX = dimY>
void setMatrix(Matrix<T, dimY, dimX> &matrix) {
  for (int i = 0; i < matrix.shape().first; ++i) {
    for (int j = 0; j < matrix.shape().second; ++j) {
      matrix[i][j] = i + j;
    }
  }
}

/****************************************************************************/
/*                                                             EXAMPLE CODE */

/* @brief ADOL-C problem
 *
 * Contains the necessary data and methods for handling an ADOL-C optimization
 * problem.
 */
struct ADProblem {
  static constexpr size_t dimIn = 3;
  static constexpr size_t dimOut = 1;

  short tapeId{-1};
  std::vector<double> inputs{dimIn, 1.0};
  std::vector<double> out{dimOut, 0.0};

  ADProblem() : tapeId(createNewTape()), inputs(dimIn, 1.0), out(dimOut, 0.0) {}
};

/* @brief Taping function
 *
 * Sets up the ADOL-C tape for automatic differentiation.
 *
 * @param problem Reference to the ADProblem.
 */
void taping(ADProblem &problem) {
  //! [contexam-recording]
  trace_on(problem.tapeId);
  {
    std::vector<adouble> indeps(problem.dimIn);
    indeps <<= problem.inputs;
    adouble result = square_sum(indeps);
    result >>= problem.out[0];
  }
  trace_off();
  //! [contexam-recording]
}

/* @brief View example
 *
 * Demonstrates the use of MatrixView to create a matrix view of a std::vector.
 */
bool viewExample() {
  //! [contexam-view]
  constexpr size_t dimX = 3;
  constexpr size_t dimY = 3;

  std::vector<double> data(dimX * dimY); // data vector that holds matrix values
  // Fill data vector with values 1, 2, ..., dimX * dimY
  for (size_t i = 0; i < dimY * dimX; ++i) {
    data[i] = static_cast<double>(i + 1);
  }
  // Create a view of the data vector in matrix shape
  std::vector<double *> view = MatrixView(data, dimX, dimY);
  //! [contexam-view]
  printVector("Data vector", data);
  printMatrix("Matrix view of data vector", view, dimX, dimY);

  for (size_t i = 0; i < dimY; i++) {
    for (size_t j = 0; j < dimX; j++) {
      if (view[i][j] != 3.0 * i + j + 1.0) {
        std::cerr << "Error: Matrix view does not match given data."
                  << std::endl;
        return false;
      }
    }
  }
  return true;
}

/* @brief Example of using ADOL-C's Matrix container
 *
 * Demonstrates the use of ADOL-C's own Matrix container with static dimensions.
 */
bool exampleMatrix() {
  //! [contexam-matrix]
  constexpr size_t dimX = 3;
  constexpr size_t dimY = 3;

  // Create an ADOL-C Matrix container with default values
  Matrix<double, dimY, dimX> matrix;

  // Show dimensions of the matrix
  std::cout << "Matrix dimensions: (" << matrix.shape().first << ", "
            << matrix.shape().second << ")\n";
  //! [contexam-matrix]

  // Print the default state of the matrix
  printMatrix("ADOL-C Matrix container", matrix);

  // Fill the matrix with a specific value
  //! [contexam-matrix-fill]
  matrix.fill(3.14);
  //! [contexam-matrix-fill]
  printMatrix("ADOL-C Matrix container after fill", matrix);

  // Give specific values to the matrix
  setMatrix(matrix);
  printMatrix("ADOL-C Matrix container after manual setting", matrix);

  if (matrix.shape().first != dimY || matrix.shape().second != dimX) {
    std::cerr << "Error: Matrix dimensions do not match expected values."
              << std::endl;
    return false;
  }
  return true;
}

/* @brief Compute using hos_forward
 *
 * Demonstrates the use of Matrix container for the hos_forward method.
 *
 * @param problem Reference to the ADProblem structure.
 */
bool computeHOS_FORWARD(ADProblem &problem) {
  //! [contexam-hos_forward]
  auto output = Matrix<double>(ADProblem::dimOut, 3);
  auto tangent = Matrix<double>(ADProblem::dimIn, 2);
  tangent[0][0] = 1.0;
  tangent[1][0] = 1.0;
  tangent[2][0] = 1.0;

  hos_forward(problem.tapeId, ADProblem::dimOut, ADProblem::dimIn, 2, 0,
              problem.inputs, tangent, problem.out, output);
  //! [contexam-hos_forward]

  printMatrix("hos_forward output", output);

  if (output[0][0] != 6.0 || output[0][1] != 3.0 || output[0][2] != 0.0) {
    return false;
  }

  return true;
}

/* @brief Compute Hessian-vector product
 *
 * Demonstrates the use of unitVector and hess_vec to compute
 * rows of the Hessian matrix for a given unit direction.
 *
 * @param problem Reference to the ADProblem structure.
 * @param unitDir Direction of the unit vector.
 */
bool computeHessianVectorProduct(ADProblem &problem, const size_t unitDir = 1) {
  //! [contexam-unitVector]
  std::vector<double> hessrow(ADProblem::dimIn);
  auto unit =
      unitVector(ADProblem::dimIn, unitDir); // unitDir has to be in [1, dimIn]

  hess_vec(problem.tapeId, ADProblem::dimIn, problem.inputs, unit, hessrow);
  //! [contexam-unitVector]

  std::string description =
      "Hessian row vector from hess_vec() for unit direction " +
      std::to_string(unitDir);
  printVector(description, hessrow);

  for (size_t j = 0; j < ADProblem::dimIn; j++) {
    if (unitDir - 1 != j && hessrow[j] != 0.0) {
      std::cerr << "Error: Hessian row vector does not match expected values."
                << std::endl;
      return false;
    } else if (unitDir - 1 == j && hessrow[j] != 2.0) {
      std::cerr << "Error: Hessian row vector does not match expected values."
                << std::endl;
      return false;
    }
  }

  return true;
}

/****************************************************************************/
/*                                                                     MAIN */

int main() {
  if (!viewExample()) {
    return 1;
  }
  if (!exampleMatrix()) { // Matrix using ADOL-C's own Matrix container, static
    return 1;             // dimensions
  }

  // Uses ADOL-C's own Matrix container, dynamic dimensions
  ADProblem problem; // Prepare ADProblem
  taping(problem);   // Perform taping of the function

  if (!computeHOS_FORWARD(problem)) {
    return 1;
  }

  if (!computeHessianVectorProduct(problem, 1) ||
      !computeHessianVectorProduct(problem, 2) ||
      !computeHessianVectorProduct(problem, 3)) {
    return 1;
  }

  return 0;
}