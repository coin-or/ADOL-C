/*----------------------------------------------------------------------------
 ADOL-C -- Automatic Differentiation by Overloading in C++
 File:     common.h
 Revision: $Id$
 Contents: Common (global) ADOL-C header

 Copyright (c) Andrea Walther, Andreas Griewank, Andreas Kowarz,
               Hristo Mitev, Sebastian Schlenkrich, Jean Utke, Olaf Vogel

 This file is part of ADOL-C. This software is provided as open source.
 Any use, reproduction, or distribution of the software constitutes
 recipient's acceptance of the terms of the accompanying license file.

----------------------------------------------------------------------------*/

#if !defined(ADOLC_COMMON_H)
#define ADOLC_COMMON_H 1
/*--------------------------------------------------------------------------*/
/* standard includes */
#if !defined(__cplusplus)
#include <stdio.h>
#include <stdlib.h>
#else
#include <cstdint>
#include <cstdlib>
#endif

/*--------------------------------------------------------------------------*/
/* type definitions */

// define bit vector propagation type used in sparse drivers
#if ADOLC_BITWORD_BITS == 32
typedef uint32_t bitword_t;
#elif ADOLC_BITWORD_BITS == 64
typedef uint64_t bitword_t;
#else
typedef uint32_t bitword_t;
#endif
typedef unsigned int uint;

/*--------------------------------------------------------------------------*/
/* system dependent configuration */
#if defined(ADOLC_INTERNAL)
#if HAVE_CONFIG_H
#include "config.h"

/*      malloc/calloc/realloc replacements */
#undef ADOLC_NO_MALLOC
#undef ADOLC_NO_REALLOC
#if !defined(HAVE_MALLOC)
#define ADOLC_NO_MALLOC 1
#else
#if (HAVE_MALLOC == 0)
#define ADOLC_NO_MALLOC 1
#endif /* HAVE_MALLOC == 0 */
#endif /* HAVE_MALLOC */
#if !defined(HAVE_REALLOC)
#define ADOLC_NO_REALLOC 1
#else
#if (HAVE_REALLOC == 0)
#define ADOLC_NO_REALLOC 1
#endif /* HAVE_REALLOC == 0 */
#endif /* HAVE_REALLOC */

#if defined(ADOLC_NO_MALLOC)
#include <adolc/rpl_malloc.h>
#define malloc rpl_malloc
#define calloc rpl_calloc
#endif /* ADOLC_NO_MALLOC */
#if defined(ADOLC_NO_REALLOC)
#include <adolc/rpl_malloc.h>
#define realloc rpl_realloc
#endif /* ADOLC_NO_REALLOC */

#ifndef HAVE_TRUNC
#define trunc(x) ((x < 0) ? ceil(x) : floor(x))
#endif

#endif /* HAVE_CONFIG_H */
#endif /* ADOLC_INTERNAL */

/*--------------------------------------------------------------------------*/
/* user parameters and settings */
#include <adolc/internal/adolc_settings.h>
#include <adolc/internal/usrparms.h>

#if defined(__cplusplus)

#define BEGIN_C_DECLS extern "C" {
#define END_C_DECLS }

#include <cassert>
#include <type_traits>
/// Helper to savely cast to a size_t. Used e.g., for short -> size_t for
/// tapeId.
template <typename T>
constexpr size_t to_size_t(T value) noexcept
  requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
{
  assert(value >= 0);
  return static_cast<size_t>(value);
}

/// Helper to savely cast to a double.
template <typename T>
constexpr double to_double(T value) noexcept
  requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
{
  return static_cast<double>(value);
}

/// Helper to evaluate a static_assert after resolving the template parameter.
template <auto> inline constexpr bool is_dependent_v = false;
#else
#define BEGIN_C_DECLS
#define END_C_DECLS
#endif

#define MAXDEC(a, b)                                                           \
  do {                                                                         \
    revreal __r = (b);                                                         \
    if (__r > (a))                                                             \
      (a) = __r;                                                               \
  } while (0)
#define MAXDECI(a, b)                                                          \
  do {                                                                         \
    int __r = (b);                                                             \
    if (__r > (a))                                                             \
      (a) = __r;                                                               \
  } while (0)
#define MINDECR(a, b)                                                          \
  do {                                                                         \
    revreal __r = (b);                                                         \
    if (__r < (a))                                                             \
      (a) = __r;                                                               \
  } while (0)
#define MINDEC(a, b)                                                           \
  do {                                                                         \
    int __r = (b);                                                             \
    if (__r < (a))                                                             \
      (a) = __r;                                                               \
  } while (0)

#define MAX_ADOLC(a, b) ((a) < (b) ? (b) : (a))
#define MIN_ADOLC(a, b) ((a) > (b) ? (b) : (a))

/*--------------------------------------------------------------------------*/
#endif
