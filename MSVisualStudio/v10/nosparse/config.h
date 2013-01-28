/* ADOL-C/src/config.h.  Generated from config.h.in by configure.  */
/* ADOL-C/src/config.h.in.  Generated from configure.ac by autoheader.  */

/* ADOL-C adouble zeroing mode */
#define ADOLC_ADOUBLE_STDCZERO 1

/* ADOL-C debug mode */
/* #undef ADOLC_DEBUG */

/* ADOL-C hard debug mode */
/* #undef ADOLC_HARDDEBUG */

/* ADOL-C Patchlevel */
#define ADOLC_PATCHLEVEL 0

/* ADOL-C Subversion */
#define ADOLC_SUBVERSION 3

/* ADOL-C thread save errno mode */
/* #undef ADOLC_THREADSAVE_ERRNO */

/* Use calloc instead of malloc for memory allocation */
#define ADOLC_USE_CALLOC 1

/* ADOL-C Version */
#define ADOLC_VERSION 2

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* Define to 1 if you have the `floor' function. */
#define HAVE_FLOOR 1

/* Define to 1 if you have the `fmax' function. */
#define HAVE_FMAX 1

/* Define to 1 if you have the `fmin' function. */
#define HAVE_FMIN 1

/* Define to 1 if you have the `ftime' function. */
#define HAVE_FTIME 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define 1 if ColPack is available */
/* #undef HAVE_LIBCOLPACK */

/* Define to 1 if your system has a GNU libc compatible `malloc' function, and
   to 0 otherwise. */
#define HAVE_MALLOC 0

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the `pow' function. */
#define HAVE_POW 1

/* Define to 1 if your system has a GNU libc compatible `realloc' function,
   and to 0 otherwise. */
#define HAVE_REALLOC 0

/* Define to 1 if you have the `sqrt' function. */
#define HAVE_SQRT 1

/* Define to 1 if stdbool.h conforms to C99. */
#define HAVE_STDBOOL_H 1

/* Define to 1 if you have the <stddef.h> header file. */
#define HAVE_STDDEF_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdio.h> header file. */
#define HAVE_STDIO_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the `strchr' function. */
#define HAVE_STRCHR 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strtol' function. */
#define HAVE_STRTOL 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/timeb.h> header file. */
#define HAVE_SYS_TIMEB_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the `trunc' function. */
/* #undef HAVE_TRUNC */

/* Define to 1 if you have the <unistd.h> header file. */
/* #undef HAVE_UNISTD_H */

/* Define to 1 if the system has the type `_Bool'. */
#define HAVE__BOOL 1

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "adolc"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "adol-c@list.coin-or.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "adolc"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "adolc 2.3.0-trunk"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "adolc"

/* Define to the home page for this package. */
#define PACKAGE_URL ""

/* Define to the version of this package. */
#define PACKAGE_VERSION "2.3.0-trunk"

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 4

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Define to 1 if you can safely include both <sys/time.h> and <time.h>. */
#define TIME_WITH_SYS_TIME 1

/* Define to 1 if your <sys/time.h> declares `struct tm'. */
/* #undef TM_IN_SYS_TIME */

/* Version number of package */
#define VERSION "2.3.0-trunk"

/* Define to empty if `const' does not conform to ANSI C. */
/* #undef const */

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to rpl_malloc if the replacement function should be used. */
#define malloc rpl_malloc

/* Define to rpl_realloc if the replacement function should be used. */
#define realloc rpl_realloc

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */
