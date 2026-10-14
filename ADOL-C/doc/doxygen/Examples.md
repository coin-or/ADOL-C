@page tutorials Tutorials

# Tutorials

Each example is compiled and checked by the documentation build.

| If you want to... | Start with... |
| --- | --- |
| Differentiate a scalar objective | @ref tutorial_gradient "1. Record and evaluate a gradient" |
| Differentiate a vector-valued function | @ref tutorial_jacobian "2. Compute a Jacobian" |
| Compute derivatives beyond first order | @ref tutorial_higher_order "3. Use higher-order scalar sweeps" |
| Avoid recording a tape | @ref tutorial_tapeless "4. Use tapeless forward mode" |
| Evaluate one tape from several threads | @ref tutorial_parallel "5. Evaluate a tape concurrently" |
| Work with piecewise-smooth functions | @ref tutorial_abs_normal "6. Build an abs-normal form" |
| Reduce memory for repeated time steps | @ref tutorial_checkpointing "7. Add checkpointing" |
| Differentiate code outside the tape interpreter | @ref tutorial_external_function "8. Add an external differentiated function" |

@anchor tutorial_gradient
## 1. Record and evaluate a gradient

**Goal:** Compute the gradient of
\f$f(x_0,x_1)=x_0x_1+\sin(x_0)\f$ at \f$(2,3)\f$.

Tape-based ADOL-C separates a computation into two phases:

1. **Record:** `trace_on()` starts a tape. The `<<=` operators mark independent
   variables, ordinary C++ expressions record operations, and `>>=` marks a
   dependent result. `trace_off()` completes the tape.
2. **Evaluate:** `gradient()` interprets that tape at the requested point. The
   same tape can be evaluated repeatedly without recording it again, provided
   the recorded control flow still represents the computation.

`createNewTape()` supplies a process-wide unique tape identifier, avoiding
hard-coded tag collisions when several components use ADOL-C.

@snippet taped_gradient.cpp taped-gradient

For this function,
\f$\nabla f=(x_1+\cos(x_0),x_0)\f$. The executable checks both entries against
that expression before printing:

@include taped_gradient.out

@anchor tutorial_jacobian
## 2. Compute a Jacobian

**Goal:** Differentiate the vector function
\f$F(x_0,x_1)=(x_0x_1,\sin(x_0)+x_1^2)\f$.

Recording multiple independent and dependent variables follows the same
pattern as the scalar case. Use the array overloads of `<<=` and `>>=` to mark all elements in order.

`jacobian()` writes an \f$m\times n\f$ matrix: one row for each dependent and
one column for each independent. Its C-compatible interface accepts `double**`,
so this example stores the values in nested `std::array` objects and passes
row pointers.

@snippet taped_jacobian.cpp taped-jacobian

The test compares all four entries with

\f[
  J_F(x)=
  \left(
  \begin{array}{cc}
    x_1 & x_0 \\
    \cos(x_0) & 2x_1
  \end{array}
  \right).
\f]

@include taped_jacobian.out

@anchor tutorial_higher_order
## 3. Use higher-order scalar sweeps

**Goal:** Compute the first three derivatives of \f$f(x)=x^4\f$ at \f$x=2\f$
with both `hos_forward()` and `hos_reverse()`.

Higher-order forward mode propagates a Taylor polynomial through
the tape. For degree \f$d\f$, each independent row contains \f$d\f$
coefficients. The input below represents

\f[
  x(t)=2+1t+0t^2+0t^3.
\f]

`hos_forward()` returns Taylor coefficients of \f$f(x(t))\f$. For the input
\f$x(t)=x+t\f$ used here, \f$Y_{k-1}=f^{(k)}(x)/k!\f$. Multiply by
\f$k!\f$ to recover the derivatives. For other input polynomials, the
coefficients describe derivatives of the composition \f$f(x(t))\f$.

A reverse sweep of degree \f$d\f$ requires a preceding forward sweep with
`keep = d + 1`. Here, `keep = 3` prepares a reverse sweep of degree two. With
\f$x(t)=x+t\f$ and output weight one, the reverse coefficients are
\f$Z_k=f^{(k+1)}(x)/k!\f$.

@snippet higher_order.cpp higher-order

The executable checks both paths against
\f$f'(2)=32\f$, \f$f''(2)=48\f$, and \f$f'''(2)=48\f$:

@include higher_order.out

@anchor tutorial_tapeless
## 4. Use tapeless forward mode

**Goal:** Compute a directional derivative while evaluating the primal
function, without recording a reusable tape.

Tapeless mode uses `adtl::adouble`. `adtl::setNumDir()` first selects the number
of derivative directions carried by every active value. `setADValue()` seeds
the direction, and each overloaded operation immediately propagates it.

Set the direction count before constructing active values and keep it fixed
until they are destroyed. Use taped mode when reverse sweeps are required.

@snippet tapeless_forward.cpp tapeless-forward

With direction \f$\dot{x}=1\f$, the derivative of
\f$x^2+\sin(x)\f$ is \f$2x+\cos(x)\f$. The executable validates both the primal
value and derivative:

@include tapeless_forward.out

@anchor tutorial_parallel
## 5. Evaluate a tape concurrently

**Goal:** Reuse one recorded scalar tape at two points on different threads.

Tape evaluation is exclusive by default and reuses owned buffers. After
recording,
`ValueTape::setSharedMode()` permits concurrent no-keep sweeps. Each task below
uses `fos_forward()` with direction one, so it obtains the function value and
first derivative independently.

Sweeps that save Taylor values require exclusive access. External
differentiated functions are unsupported in shared no-keep sweeps. This example
uses low-level sweeps; shared mode does not make all high-level drivers
concurrent. Drivers that cache tape-owned scratch data need separate tapes or
application-level synchronization.

@snippet parallel_evaluation.cpp parallel-evaluation

Both tasks are joined before their results are checked against
\f$2x+\cos(x)\f$:

@include parallel_evaluation.out

@anchor tutorial_abs_normal
## 6. Build an abs-normal form

**Goal:** Extract an abs-normal representation of the piecewise-smooth
function \f$y=x_0+x_1-|x_0|-|x_1|\f$.

Record the absolute-value operations, then query the switch count.
`enableMinMaxUsingAbs()` also represents min/max operations as absolute-value
switches; this example uses `fabs()` directly.

@snippet abs_normal_struct.cpp abs-normal-recording

`ADOLC::AbsNormalForm` owns its matrix storage.
`ADOLC::AbsNormalForm::fromTape()` obtains the dimensions from the tape, and
`ADOLC::abs_normal()` fills the \f$L\f$, \f$Z\f$, \f$Y\f$, and \f$J\f$ blocks.

@snippet abs_normal_struct.cpp abs-normal-struct

For this two-switch example, the executable verifies the shape and every
matrix coefficient:

@include abs_normal_struct.out

@anchor tutorial_checkpointing
## 7. Add checkpointing to repeated time steps

**Goal:** Differentiate 100 Euler steps without keeping the complete unrolled
time-stepping tape.

Checkpointing trades recomputation for memory. The step function is available
for both `adouble` and `double`; `ADOLC::CP::Context` describes how many steps
to execute, how many checkpoints may be retained, and where the state enters
and leaves the checkpointed region.

The state variables must occupy consecutive tape locations. Call
`ensureContiguousLocations()` before constructing them, as shown below.

@snippet checkpointing.cpp checkpointing-context

The complete-loop tape provides a reference implementation. Both approaches
must reproduce the analytic gradient
\f$(1.01^{100},1.02^{100})\f$:

@include checkpointing.out

@anchor tutorial_external_function
## 8. Add an external differentiated function

**Goal:** Record each Euler step as an external operation with separate
derivative callbacks.

Derive from `EDFobject` to supply primal and derivative callbacks. This
example implements them in two ways:

- a nested implementation delegates the callbacks to a small inner ADOL-C
  tape;
- a manual implementation evaluates the primal and first-order formulas
  directly.

The object must outlive every evaluation of the outer tape that refers to it.
Implement each callback requested by the application and check its result
against full taping.

For the Euler step, the scalar forward callback first evaluates the primal
function and then applies the Jacobian to the incoming direction:

@snippet edfootest.cpp edf-manual-forward

The scalar reverse callback applies the transposed Jacobian to the incoming
adjoints. It adds to `z` because operations visited earlier in the reverse
sweep may already have contributed:

@snippet edfootest.cpp edf-manual-reverse

The outer tape can then use either the nested or manual implementation through
the same object interface:

@snippet edfootest.cpp edf-object

The executable compares full taping, nested ADOL-C, and manual callbacks with
the same analytic time-step gradient:

@include edf_object.out

## Build the examples

Configure with Doxygen and Graphviz installed, then run:

```shell
cmake -S . -B build-docs -DCMAKE_BUILD_TYPE=Release
cmake --build build-docs --target docs -j
```

CMake runs each example in a separate working directory and includes its output
only after its result checks pass. Add new examples to
`ADOL-C/doc/doxygen/CMakeLists.txt` with a snippet region, expected results,
a nonzero exit status on failure, and stable output.
