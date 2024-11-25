# ADOL-C: A Package for the Automatic Differentiation of Algorithms Written in C/C++

## Scope of the library

ADOL-C is a library for automatic differentiation written in C++
featuring e.g.

- traceless-forward mode
- tape-based mode
- sparse Jacobians and sparse Hessians
- external differentiated functions
- optimal checkpointing
- adapted differentiation of fixed point iterations
- parallel differentiation of OpenMP-parallel loops
- Lie derivatives of scalar, vector and covector fields


## Documentation


### Class documentation

The library contains a class documentation which can be build using [doxygen].
You can build the documentation using
```shell
doxygen Doxyfile
```
from within thd `doc/` subdirectory.
For an overview of the contained features and submodules
you can e.g. refer to the [Topics](topics.html) section of the doxygen documentation.

@todo Add support for building the class documentation with cmake.

### Manual

A manual of ADOL-C is contained in the `ADOL-C/doc/` directory
and was also published as

@todo Add reference to manual.

### Examples

@todo Describe where to find examples.



## Using ADOL-C and licensing

@todo How to incorporate ADOL-C in user code.
@todo How to cite ADOL-C
@todo Licence



## Building ADOL-C

### Dependencies
### Building the library

[doxygen]: http://www.stack.nl/~dimitri/doxygen/
