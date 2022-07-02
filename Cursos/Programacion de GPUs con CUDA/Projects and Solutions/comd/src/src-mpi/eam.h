/// \file
/// Compute forces for the Embedded Atom Model (EAM).

#ifndef __EAM_H
#define __EAM_H

#include "mytype.h"

struct BasePotentialSt;
struct LinkCellSt;



#ifdef __cplusplus
extern "C" {
#endif

        void lunch_eamForce_kernel(float *rhobar, float *dfEmbed, float phi_n, float phi_x0, float phi_invDx, float *phi_values,
	                                                          float rho_n, float rho_x0, float rho_invDx, float *rho_values,
                                                                  float f_n, float f_x0, float f_invDx, float *f_values,
						                  int nLB, int nTB, int nHB, int *nA, int *gridS, float *rr, float *Uu, float *ff,
						                  float Cutoff, float *etot);


        void lunch_eamForce_kernel_2(float *dfEmbed, float rho_n, float rho_x0, float rho_invDx, float *rho_values,
                                    int nLB, int nTB, int nHB, int *nA, int *gridS, float *rr, float *ff, float Cutoff);


#ifdef __cplusplus
}
#endif





/// Pointers to the data that is needed in the load and unload functions
/// for the force halo exchange.
/// \see loadForceBuffer
/// \see unloadForceBuffer
typedef struct ForceExchangeDataSt
{
   real_t* dfEmbed; //<! derivative of embedding energy
   struct LinkCellSt* boxes;
}ForceExchangeData;

struct BasePotentialSt* initEamPot(const char* dir, const char* file, const char* type);


#endif
