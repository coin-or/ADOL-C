@page developer_guide ADOL-C developer guide

# ADOL-C developer guide

This guide introduces the internals of ADOL-C for developers who want to
contribute to the library. For mathematical background and detailed API, the @ref manual remains the best source. Coding conventions, build
instructions, and the contribution workflow are documented in
`CONTRIBUTING.md`.

## The central tape-based model

The model is based on `adouble`, which is primarily a `tape_location` with a
double-like interface that records operations. A `tape_location` represents a
location in the active value store (`globalTapeVars.store`) and has a defined
ownership model that enables move semantics.

Tape-based ADOL-C has two distinct phases:

1. **Recording:** overloaded `adouble` operations register corresponding
   opcodes, locations, passive values, and possible Taylor coefficients on the
   `ValueTape`.
2. **Evaluation:** a forward or reverse sweep interprets the recorded
   information at user-provided inputs, derivative directions, or adjoint
   weights.

A tape records several synchronized buffers in `TapeRecordingContext`:

| Buffer | Contents | Main abstraction |
| --- | --- | --- |
| operation buffer | opcodes such as multiplication or sine | `OpInfo` |
| location buffer | active input and result locations used by an opcode | `LocInfo` |
| value buffer | passive constants, separate from the active-value store | `ValInfo` |
| Taylor buffer | saved Taylor coefficients needed by reverse mode | `TayInfo` |

The buffers may remain in memory or spill to files in fixed-size blocks. Their
metadata records where each forward or reverse sweep must find the next block.

## Repository map

| Path | Content | Good starting points |
| --- | --- | --- |
| `ADOL-C/include/adolc/adolc.h` | the public umbrella header | overview of exported features |
| `ADOL-C/include/adolc/tape_interface.h` | tape API | `trace_on`, `trace_off` |
| `ADOL-C/include/adolc/adtb_types.h` | types that record active operations | `tape_location`, `adouble` |
| `ADOL-C/src/adouble.cpp` | taped scalar operations and recording logic | assignment and arithmetic operators |
| `ADOL-C/src/uni5_for.cpp` | generates zero-, first-, vector-, and higher-order forward interpreters such as `zos_forward.cpp` | macro mechanics and the handling of `plus_a_a` |
| `ADOL-C/src/fo_rev.cpp` | generates first-order scalar and vector reverse interpreters | Taylor buffer usage in `mult_d_a` |
| `ADOL-C/src/ho_rev.cpp` | generates higher-order reverse interpreters such as `hos_reverse.cpp` | Taylor buffer usage and recurrence relation in `mult_a_a` |
| `ADOL-C/include/adolc/valuetape` | tape ownership, buffers, contexts, registry, and synchronization | `valuetape.h` (e.g., `ValueTape::init_sweep`), `bufferstate.h`, `taperegistry.h` |
| `ADOL-C/src/drivers` | convenience algorithms built on the generated low-level sweeps | `gradient`, `hessian`, `tensor_eval` |
| `ADOL-C/src/sparse` | (ColPack-backed) sparse recovery | `sparse_jac` |
| `ADOL-C/src/externfcts*.cpp` | external differentiated function support | `reg_ext_fct`, `call_ext_fct` |
| `ADOL-C/boost-test` | unit, regression, integration, sparse, and thread-safety tests | tests closest to the methods of interest |
| `ADOL-C/doc/doxygen` | generated API pages and compiled documentation tutorials | this guide and @ref tutorials |

## Recording lifecycle

A recording follows this sequence:

1. `createNewTape()` inserts a `ValueTape` into the process-wide registry and
   returns its unique ID. If the thread has no current tape, the new tape becomes
   current.
2. Active values allocate locations from the current tape. Their construction
   and destruction must occur while the correct tape is current.
3. `trace_on()` starts recording and pushes a `RecordingFrame` onto the
   thread-local current-tape stack.
4. Independent markers, operators in `adouble.cpp`, and dependent markers
   append opcodes, locations, and passive values through `ValueTape`.
5. `trace_off()` finalizes statistics and restores the previous current tape.

The registry owns tapes process-wide, but the selected tape and nested
recording stack are thread-local. This distinction is important: tape lookup by
ID and tape selection through `currentTape()` behave differently.

## Evaluation lifecycle

Low-level sweep names encode their shape:

- `zos`: zero-order scalar forward;
- `fos` / `fov`: first-order scalar/vector;
- `hos` / `hov`: higher-order scalar/vector;
- `_forward` and `_reverse`: traversal and derivative accumulation direction.

High-level routines such as `gradient()` and `jacobian()` validate dimensions
and compose these sweeps. When debugging a driver, first determine which
low-level sweeps it invokes and whether it requests `keep`.

Every interpreter calls `ValueTape::init_sweep()` and receives a
`TapeEvaluationContext`. The context owns independent cursors for the tape
buffers and closes/releases resources at the end of the sweep. Forward
interpreters read from the beginning; reverse interpreters prepare buffers at
the end and walk backwards through them.

`keep` controls whether a forward sweep saves Taylor coefficients for a later
reverse pass. If used, updating a `pdouble` parameter invalidates saved Taylor
coefficients, so a new forward sweep with `keep >= 1` is required before reverse
evaluation.

## Buffer ownership and thread safety

`TapeRecordingContext` owns the buffers after recording.
`BufferState` can either own its allocation or be a non-owning view with an
independent cursor. Views do not track or invalidate themselves when their
owner moves, reallocates, or dies; the surrounding tape locks guarantee their
lifetime.

During evaluation, for example in `fos_forward`, the default exclusive mode
takes exclusive access and temporarily moves reusable buffers into the
evaluation context; see `ValueTape::init_sweep()`. Ownership is returned to
`TapeRecordingContext` at the end of the sweep by `ValueTape::end_sweep()`.
This retains the single-threaded fast path.

After `ValueTape::setSharedMode()`, concurrent no-keep sweeps take shared
access to the buffers. Each evaluation initially views the canonical buffers
through independent cursors and allocates an owned buffer lazily only when it
must load or overwrite data. Keep sweeps still
require exclusive access because they publish Taylor data. External
differentiated functions are currently rejected in shared mode.

When modifying this area, preserve these invariants:

- modification of the `ValueTape::DataAccess` mode cannot overlap recording or
  evaluation;
- an owner outlives every buffer view;
- shared sweeps never mutate tape data;
- only an exclusive sweep publishes Taylor data or buffers;
- exception paths release locks and return moved buffers to the recording
  context.

## Adding or changing an opcode

New functionality or operation fusion can require an additional opcode. Search
for a similar opcode across the source tree before starting. Existing
operations reveal the operand ordering and sweep-specific conventions. Please follow
those conventions.

An opcode change is cross-cutting. Use this checklist:

1. Define the opcode in `adolc/oplate.h` without silently changing the meaning
   of existing serialized values.
2. Record it from the relevant `adouble` or `pdouble` operation, including
   locations and any passive values.
3. Implement primal handling for every applicable forward interpreter.
4. Implement derivative propagation in all supported first- and higher-order
   forward modes.
5. Handle `keep >= 1` as preparation for reverse passes.
6. Implement adjoint propagation in all supported reverse modes.
7. Update sparsity, piecewise-linear, tapeless, tape-doc, and
   external-function paths when the operation is relevant to them.
8. Compare tape statistics and operation counts so a convenient overload does
   not accidentally record a longer sequence.

## Debugging and performance

`printTapeStats(tapeId)` and `tapestats(tapeId)` expose counts for operations,
locations, values, buffer sizes, and file access. `tape_doc()` can print the
interpreted operation sequence. These tools are useful for distinguishing a
recording regression from a slower interpreter.
