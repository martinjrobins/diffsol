#include <sundials/sundials_version.h>
#if SUNDIALS_VERSION_MAJOR > 5
#include "idaRoberts_dns_v6.c"
#else
#include "idaRoberts_dns_v5.c"
#endif