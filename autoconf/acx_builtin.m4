AC_DEFUN([ACX_C_BUILTIN_EXPECT],[
    AC_MSG_CHECKING(for __builtin_expect)
    AC_LANG(C)
    AC_TRY_LINK(
      [
          int f(){return 1;};
      ],
      [
          if (__builtin_expect(f(), 0));
      ],
      [
        AC_MSG_RESULT(yes)
        AC_DEFINE(HAVE_BUILTIN_EXPECT, 1,
                  [Define if the compiler provides __builtin_expect])
      ],
      [
        AC_MSG_RESULT(no)
      ])
])