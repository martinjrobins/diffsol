#include "mangle.h"
#define SIZE 20 
#define FUNC_NAME MANGLE(idaHeat2d_klu, SIZE)

#include <sundials/sundials_version.h>
#if SUNDIALS_VERSION_MAJOR > 5
#include "idaHeat2d_klu_v6.c"
#else
#include "idaHeat2d_klu_v5.c"
#endif


