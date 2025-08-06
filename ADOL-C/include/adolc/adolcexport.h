#ifndef ADOLC_EXPORT_H
#define ADOLC_EXPORT_H

/* Under windows we have to explicit state which symbols (functions, classes,
 * etc) are exportet when compiling a SHARED library. The dllimport specifies if
 * a user builds its code with adolc that the symbols are defined somewhere
 * else.
 */
#if defined(_WIN32) && defined(ADOLC_SHARED)
#ifdef BUILD_ADOLC
#define ADOLC_API __declspec(dllexport)
#else
#define ADOLC_API __declspec(dllimport)
#endif
#else
#define ADOLC_API
#endif

#endif // ADOLC_EXPORT_H