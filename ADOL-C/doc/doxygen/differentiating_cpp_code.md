@page differentiating_cpp_code Differentiating C++ Code

## Differentiating C++ Code

ADOL-C utilizes C++'s operator overloading features to compute derivatives of programs. This operator overloading is applied via ADOL-C's main datatype `adouble`. Basic derivative evaluation can be done using tape-based or tape-less AD.

The tape-based AD utilizes a tape to store intermediate values to be later used to compute, e.g., a gradient using reverse mode AD or many of the derivative drivers. In contrast, the tape-less `adouble` is only capable of forward-mode computations, but serves as a light-weight option, which can be much faster in some situations.

---

## Tape-based AD

To compute derivatives using tape-based AD, all necessary operations must be included in a [`trace_on`](@ref trace_on) – [`trace_off`](@ref trace_off) block.

Inside this block, all operations involving `adouble` variables are traced onto the tape.

> 💡 **Best practice**: allocate and free all `adouble` objects within the same [`trace_on`](@ref trace_on) – [`trace_off`](@ref trace_off) block.

---

## Tape-less AD

In comparison, for tape-less AD ...

*(To be completed similarly)*

---

## Troubleshooting

### `adouble` variables outside [`trace_on`](@ref trace_on) – [`trace_off`](@ref trace_off) block

It is possible (though not recommended) to have `adouble` variables outside the tracing block. To do so, you must explicitly create a tape using [`createNewTape`](@ref createNewTape) and select it using [`setCurrentTape`](@ref setCurrentTape) **before** allocating any `adouble` on this specific tape.

> ⚠️ When working with multiple tapes, be careful during deallocation:
> `adouble` objects notify their associated tape via [`currentTape`](@ref currentTape) during destruction. If the wrong tape is selected, this may result in undefined behavior or errors.

However, it is always safe to encapsulate logic like this:

```cpp
const short tapeId = createNewTape();
call_your_function(tapeId);
gradient(tapeId, ...);
```

with

```cpp
void call_your_function(short tapeId) {
    setCurrentTape(tapeId);

    // Allocate your data here
    std::array<adouble, 10> data;

    trace_on(tapeId);

    // Computations involving `adouble`

    trace_off();
}
```
