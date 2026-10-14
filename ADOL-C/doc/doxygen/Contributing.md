@page contribution_guide Contributing to ADOL-C

# Contributing to ADOL-C

Read the @ref developer_guide before, and search
 existing issues and pull requests before starting. Discuss API changes,
new dependencies, and tape-format changes in an issue first. Keep unrelated
cleanup in separate pull requests.

## Build and test

ADOL-C requires CMake 3.19+ and a C++20 compiler: GCC 11+, Clang 13+, or
MSVC 19.31+. Tests require Boost and OpenMP. Sparse builds require ColPack and
OpenMP; CMake fetches ColPack unless `ColPack_DIR` selects an installed package.

```shell
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DBUILD_INTERFACE=ON \
  -DENABLE_SPARSE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Omit `BUILD_INTERFACE` and `ENABLE_SPARSE` during development if the change
does not affect them. Check the relevant optional builds before submitting.

List Boost.Test cases or run one case or suite:

```shell
build/ADOL-C/boost-test/boost-test-adolc --list_content
build/ADOL-C/boost-test/boost-test-adolc --run_test='suite-or-case-name'
```

For memory checks with GCC or Clang:

```shell
cmake -S . -B build-asan \
  -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTS=ON \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure
```

The pull request test workflow also runs Valgrind.

## Documentation

Document public APIs in the headers with Doxygen comments. State dimensions,
ownership, preconditions, and threading restrictions. Avoid comments that repeat the code.

Install Doxygen and Graphviz, then build:

```shell
cmake -S . -B build-docs -DCMAKE_BUILD_TYPE=Release
cmake --build build-docs --target docs -j
```

The `docs` target compiles and runs each displayed example before generating
HTML. Examples must check their results, return nonzero on failure, and print
stable output. Register new examples in `ADOL-C/doc/doxygen/CMakeLists.txt`.
Keep `Doxyfile.in` in Doxygen's generated layout.


## C++ conventions

Use these conventions for new or substantially rewritten code. Preserve public
API names and avoid unrelated legacy rewrites. 

| Entity | Convention | Example |
| --- | --- | --- |
| types and template type parameters | `UpperCamelCase` | `TapeEvaluationContext`, `BufferType` |
| functions, variables, and parameters | `lowerCamelCase` | `createNewTape()`, `tapeId` |
| private data members | `lowerCamelCase_` | `innerTapeId_` |
| constants and scoped-enum values | `lowerCamelCase` | `reverseDegree` |
| macros | `UPPER_SNAKE_CASE` | `CURRENT_LOCATION` |

Use established abbreviations such as `fos`, `hov`, `loc`, and `tay` where
appropriate. Otherwise, choose descriptive names. You might encounter legacy code
that does not follow these instructions.

- Treat raw pointers as non-owning unless the interface documents ownership.
  State whether public pointer arguments may be null.
- Use references for required objects and `std::span` for borrowed ranges.
  Retain pointer-plus-dimension APIs where compatibility requires them.
- Use `std::unique_ptr` for exclusive ownership and `std::shared_ptr` for shared
  ownership if it does not hurt the performance significantly.
  Manage memory, files, tape state, and locks through RAII.
- Use `constexpr`, concepts, templates, and static dispatch when they remove
  repeated work or prevent invalid states. Keep simple run-time code when it
  is clearer and does not hurt performance significantly. Measure changes to recording and sweep loops.
- Use `if constexpr` for type-dependent branches and `enum class` for states.
  Avoid boolean arguments if possible.
- Prefer standard containers and algorithms when tape layout and performance
  permit them. Make immutability explicit with `const`.
- Use `noexcept` only when the function cannot throw.
- Check numerical conversions and dimensions. Distinguish inputs, outputs,
  directions, degrees, and coefficient counts.
- Keep public headers self-contained.

## Formatting and static analysis

The formatting workflow uses clang-format 19.1.3. Format changed C/C++ files:

```shell
clang-format -i path/to/changed_file.cpp path/to/changed_header.h
```

Do not run clang-format on Markdown, CMake, or Doxygen configuration files.
Review formatting changes before committing.

Enable clang-tidy for the library target with:

```shell
cmake -S . -B build-tidy \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_CLANG_TIDY=ON
cmake --build build-tidy -j
```

The codebase has existing clang-tidy warnings. Avoid new warnings in changed
files and keep broader cleanup separate.

## Tests

| Area | Location |
| --- | --- |
| Taped operators | `ADOL-C/boost-test/traceOperator*.cpp` |
| Tapeless operators | `ADOL-C/boost-test/traceless*.cpp` |
| Tape ownership, buffers, concurrency | `ADOL-C/boost-test/valuetape` |
| Sparse recovery | `ADOL-C/boost-test/sparse` |
| Higher-order reverse | `ADOL-C/boost-test/ho_rev` |
| Documentation examples | `ADOL-C/doc/doxygen/examples` |
| Package examples | `ADOL-C/examples` |

Add a regression test for each bug fix. Cover every supported forward and
reverse mode for new active operations. Tape and buffer changes need 
in-memory, file-backed, ownership, and exception coverage. Use deterministic 
synchronization in concurrency tests. Sparse changes need pattern and numerical recovery checks.

ADOL-C treats compiler warnings as errors. Test Debug and Release when
initialization, assertions, or optimization affect the result.


## Before submitting

- [ ] Tests and relevant optional builds pass.
- [ ] Changed C++ files pass clang-format.
- [ ] The documentation builds for documentation and public API changes.
- [ ] API, ownership, and threading changes are documented.
- [ ] Performance claims have a reproducible baseline.
- [ ] The diff contains no unrelated files, generated output, or build artifacts.

Describe the problem, resulting behavior, validation, and compatibility changes
in the pull request.
