/// \file
/// Computes forces for the 12-6 Lennard Jones (LJ) potential.

#ifndef _LJTYPES_H_
#define _LJTYPES_H_

#include "CoMDTypes.h"

struct BasePotentialSt;
struct BasePotentialSt* initLjPot(void);


#ifdef __cplusplus
extern "C" {
#endif

void lunch_ljForce_kernel(real_t *p, int nLB, int *nA, int *gridS, int *g, real3 *rr, real_t *Uu, real3 *ff, int sz);


#ifdef __cplusplus
}
#endif


#endif

