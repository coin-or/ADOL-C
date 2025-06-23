#include <adolc/internal/adolc_settings.h>

const ADOLC_ID &get_adolc_id() {
  // Tape identification (ADOLC & version check)
  static ADOLC_ID adolc_id{
      ADOLC_VERSION,  ADOLC_SUBVERSION, ADOLC_PATCHLEVEL,
      sizeof(size_t), sizeof(double),   sizeof(size_t),
  };
  return adolc_id;
}