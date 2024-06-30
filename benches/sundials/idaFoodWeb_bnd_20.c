#include "mangle.h"
#define SIZE 20
#define FUNC_NAME MANGLE(idaFoodWeb_bnd, SIZE)

#include <sundials/sundials_version.h>
#if SUNDIALS_VERSION_MAJOR > 5
#include "idaFoodWeb_bnd_v6.c"
#else
#include "idaFoodWeb_bnd_v5.c"
#endif


