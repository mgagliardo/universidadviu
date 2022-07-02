/* eam_cu.cu */
#include <stdio.h>
#include <math.h>
#include "eam.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

#define MAXATOMS 32


__constant__ int nLB_c[1];
__constant__ int nTB_c[1];
__constant__ int nHB_c[1];
//__constant__ int nA_c[216]; //works for now!!!!! ----> should be number of local boxes nLB
__constant__ int gridS_c[3]; //number of boxes in each dimension on processor
__constant__ float Cutoff_c[1];

__constant__ float phi_n_c[1];
__constant__ float phi_x0_c[1];
__constant__ float phi_invDx_c[1];
//phi_values cannot be made constant as we cannot know how big it is until we read a file

__constant__ float rho_n_c[1];
__constant__ float rho_x0_c[1];
__constant__ float rho_invDx_c[1];
//rho_values cannot be made constant as we cannot know how big it is until we read a file


__constant__ float f_n_c[1];
__constant__ float f_x0_c[1];
__constant__ float f_invDx_c[1];
//f_values cannot be made constant as we cannot know how big it is until we read a file


//helper functions
__device__ int getBoxFromTuple(int ix, int iy, int iz)
{
   int iBox = 0;
   //const int* gridS_c = boxes->gridS_c; // alias

   //printf("grids_c[0] = %d, grids_c[1] = %d, grids_c[2] = %d \n",  gridS_c[0],  gridS_c[1],  gridS_c[2]);

   // Halo in Z+
   if (iz == gridS_c[2])
   {
      iBox = nLB_c[0] + 2*gridS_c[2]*gridS_c[1] + 2*gridS_c[2]*(gridS_c[0]+2) +
         (gridS_c[0]+2)*(gridS_c[1]+2) + (gridS_c[0]+2)*(iy+1) + (ix+1);
   }
   // Halo in Z-
   else if (iz == -1)
   {
      iBox = nLB_c[0] + 2*gridS_c[2]*gridS_c[1] + 2*gridS_c[2]*(gridS_c[0]+2) +
         (gridS_c[0]+2)*(iy+1) + (ix+1);
   }
   // Halo in Y+
   else if (iy == gridS_c[1])
   {
      iBox = nLB_c[0] + 2*gridS_c[2]*gridS_c[1] + gridS_c[2]*(gridS_c[0]+2) +
         (gridS_c[0]+2)*iz + (ix+1);
   }
   // Halo in Y-
   else if (iy == -1)
   {
      iBox = nLB_c[0] + 2*gridS_c[2]*gridS_c[1] + iz*(gridS_c[0]+2) + (ix+1);
   }
   // Halo in X+
   else if (ix == gridS_c[0])
   {
      iBox = nLB_c[0] + gridS_c[1]*gridS_c[2] + iz*gridS_c[1] + iy;
   }
   // Halo in X-
   else if (ix == -1)
   {
      iBox = nLB_c[0] + iz*gridS_c[1] + iy;
   }
   // local link celll.
   else
   {
      iBox = ix + gridS_c[0]*iy + gridS_c[0]*gridS_c[1]*iz;
   }
   //assert(iBox >= 0);
   if(iBox < 0)
       printf("whaaaaaaaaaaaaaaaaaaaaaaaaaaaa \n");
   //assert(iBox < nTB_c[0]);  //look at this later!!!!!
   //if (iBox >= nTB_c[0])
       //printf("nooooooooooooooooooooooooooooo \n");
    //printf("iBox: %d, nLB_c[0] = %d, gridS_c = (%d, %d, %d)\n", iBox, nLB_c[0], gridS_c[0], gridS_c[1], gridS_c[2]);
   return iBox;
}



__device__ void getTuple(int iBox, int* ixp, int* iyp, int* izp)
{
   int ix, iy, iz;
   //const int* gridS_c = gridS_c[0]; // alias

   // If a local box
   if( iBox < nLB_c[0])
   {
      ix = iBox % gridS_c[0];
      iBox /= gridS_c[0];
      iy = iBox % gridS_c[1];
      iz = iBox / gridS_c[1];
   }
   // It's a halo box
   else
   {
      int ink;
      ink = iBox - nLB_c[0];
      if (ink < 2*gridS_c[1]*gridS_c[2])
      {
         if (ink < gridS_c[1]*gridS_c[2])
         {
            ix = 0;
         }
         else
         {
            ink -= gridS_c[1]*gridS_c[2];
            ix = gridS_c[0] + 1;
         }
         iy = 1 + ink % gridS_c[1];
         iz = 1 + ink / gridS_c[1];
      }
      else if (ink < (2 * gridS_c[2] * (gridS_c[1] + gridS_c[0] + 2)))
      {
         ink -= 2 * gridS_c[2] * gridS_c[1];
         if (ink < ((gridS_c[0] + 2) *gridS_c[2]))
         {
            iy = 0;
         }
         else
         {
            ink -= (gridS_c[0] + 2) * gridS_c[2];
            iy = gridS_c[1] + 1;
         }
         ix = ink % (gridS_c[0] + 2);
         iz = 1 + ink / (gridS_c[0] + 2);
      }
      else
      {
         ink -= 2 * gridS_c[2] * (gridS_c[1] + gridS_c[0] + 2);
         if (ink < ((gridS_c[0] + 2) * (gridS_c[1] + 2)))
         {
            iz = 0;
         }
         else
         {
            ink -= (gridS_c[0] + 2) * (gridS_c[1] + 2);
            iz = gridS_c[2] + 1;
         }
         ix = ink % (gridS_c[0] + 2);
         iy = ink / (gridS_c[0] + 2);
      }

      // Calculated as off by 1
      ix--;
      iy--;
      iz--;
   }

   *ixp = ix;
   *iyp = iy;
   *izp = iz;
}


__device__ void interpolate_phi(float* tt, float r, float* f, float* df)
{
   //const real_t* tt = table->values; // alias
   //printf("phi_x0_c = %f, phi_invDx_c = %f, phi_n_c = %f\n", phi_x0_c[0], phi_invDx_c[0], phi_n_c[0]);
   if ( r < phi_x0_c[0] ) r = phi_x0_c[0];

   r = (r - phi_x0_c[0])*(phi_invDx_c[0]) ;
   int ii = (int)floor(r);
   if (ii > phi_n_c[0])
   {
      ii = phi_n_c[0];
      r = phi_n_c[0] / phi_invDx_c[0];
   }
   // reset r to fractional distance
   r = r - floor(r);

   // real_t g1, g2;
//we had commented this two lines before

   //if (ii-1 < 0)
   //	printf("tt[%d] = %f, tt[%d] = %f \n",  ii+1, tt[ii+1], ii-1, tt[ii-1]);
   

   float g1 = tt[ii+1] - tt[ii-1];
   float g2 = tt[ii+2] - tt[ii];

   *f = tt[ii] + 0.5*r*(g1 + r*(tt[ii+1] + tt[ii-1] - 2.0*tt[ii]) );

   

   *df = 0.5*(g1 + r*(g2-g1))*phi_invDx_c[0];
}



__device__ void interpolate_rho(float* tt, float r, float* f, float* df)
{
   //const real_t* tt = table->values; // alias
   //printf("rho_x0_c = %f, rho_invDx_c = %f, rho_n_c = %f\n", rho_x0_c[0], rho_invDx_c[0], rho_n_c[0]);

   if ( r < rho_x0_c[0] ) r = rho_x0_c[0];

   r = (r - rho_x0_c[0])*(rho_invDx_c[0]) ;
   int ii = (int)floor(r);
   if (ii > rho_n_c[0])
   {
      ii = rho_n_c[0];
      r = rho_n_c[0] / rho_invDx_c[0];
   }
   // reset r to fractional distance
   r = r - floor(r);
   //if(ii-1 < 0)
   //	printf("tt[%d] = %f, tt[%d] = %f \n",  ii+1, tt[ii+1], ii-1, tt[ii-1]);


   float g1 = tt[ii+1] - tt[ii-1];
   float g2 = tt[ii+2] - tt[ii];

   *f = tt[ii] + 0.5*r*(g1 + r*(tt[ii+1] + tt[ii-1] - 2.0*tt[ii]) );

   *df = 0.5*(g1 + r*(g2-g1))*rho_invDx_c[0];
}


__device__ void interpolate_f(float* tt, float r, float* f, float* df)
{
   //const real_t* tt = table->values; // alias
   //printf("f_x0_c = %f, f_invDx_c = %f, f_n_c = %f\n", f_x0_c[0], f_invDx_c[0], f_n_c[0]);
   if ( r < f_x0_c[0] ) r = f_x0_c[0];

   r = (r - f_x0_c[0])*(f_invDx_c[0]) ;
   int ii = (int)floor(r);
   if (ii > f_n_c[0])
   {
      ii = f_n_c[0];
      r = f_n_c[0] / f_invDx_c[0];
   }
   // reset r to fractional distance
   r = r - floor(r);
   //if(ii-1 < 0)
   //{
	//printf("f_x0_c = %f, f_invDx_c = %f, f_n_c = %f, r = %f\n", f_x0_c[0], f_invDx_c[0], f_n_c[0], r);
   	//printf("tt[%d] = %f, tt[%d] = %f \n",  ii+1, tt[ii+1], ii-1, tt[ii-1]);
   //}   
//if(ii == 0) ++ii;

   float g1 = tt[ii+1] - tt[ii-1];
    
   float g2 = tt[ii+2] - tt[ii];

   *f = tt[ii] + 0.5*r*(g1 + r*(tt[ii+1] + tt[ii-1] - 2.0*tt[ii]) );

   *df = 0.5*(g1 + r*(g2-g1))*f_invDx_c[0];
}




__device__ int getNeighborBoxes(int iBox, int* nbrBoxes)
{
    int ix, iy, iz;
    getTuple(iBox, &ix, &iy, &iz);

    unsigned count = 0;
    for (int i=ix-1; i<=ix+1; i++)
        for (int j=iy-1; j<=iy+1; j++)
            for (int k=iz-1; k<=iz+1; k++){
                    //if (count >= 27)
                    //    printf("WTF YO!! ix=%d, iy=%d, iz=%d\n", ix, iy, iz);
                    //printf("count = %u, iBox = %d, nbrBoxes = %p\n", count, iBox, nbrBoxes); 
                    nbrBoxes[count++] = getBoxFromTuple(i,j,k);
             }
    return count;
}


/*__device__ double atomicAdd(double* address, double val)
{
    //printf("IN: threadIdx = %d, address = %f, valu = %f\n", threadIdx.x, *address, val);
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    //printf("OUT: threadIdx = %d, address = %f, val=%f\n", threadIdx.x, *address, val);
    return __longlong_as_double(old);
}


__device__ double atomicSub(double* address, double val)
{
    //printf("val = %f\n", val);
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(
                        __longlong_as_double(assumed) - val));
    } while (assumed != old);
    return __longlong_as_double(old);
}*/

//end of helper functions



//eam third pass kernel or eam2 kernel
__global__ void eamForce_2(float *dfEmbed_d, float *rho_values_d, int *nA_d, float *rr_d, float *ff_d)
{

    __shared__ int nbrBoxes[27];

     int tx = threadIdx.x;
    int iBox = blockIdx.x;

    __shared__ int nIBox;
    __shared__ int nNbrBoxes;


    if (tx == 0)
    {
	//number of atoms in this box
        nIBox = nA_d[iBox];
	//number of neighboring boxes
        nNbrBoxes = getNeighborBoxes(iBox, nbrBoxes);

    }

    __syncthreads();


    //algorithm
    if(tx < nIBox)
    {
	int iOff = MAXATOMS * iBox + tx;

        float rr_dloc[3];
        rr_dloc[0] = rr_d[(iOff*3)];
        rr_dloc[1] = rr_d[(iOff*3) + 1];
        rr_dloc[2] = rr_d[(iOff*3) + 2];

        // loop over neighbor boxes of iBox (some may be halo boxes)
        for (int jTmp=0; jTmp<nNbrBoxes; jTmp++)
        {
	    int jBox = nbrBoxes[jTmp];
            if (jBox < iBox ) continue;

            int nJBox = nA_d[jBox];  //<-------- this can be done faster!!!!!!

            // loop over atoms in jBox
            for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++)
	    {
		if ( (iBox==jBox) &&(ij <= tx) ) continue;

                double r2 = 0.0;
                float dr[3];
                for (int k=0; k<3; k++)
                {
                    dr[k]= rr_dloc[k] - rr_d[(jOff*3)+k];  //<--------- this can be done faster!!!!
                    r2+=dr[k]*dr[k];
                }

                if(r2>Cutoff_c[0]) continue;

		double r = sqrt(r2);

                float rhoTmp, dRho;
                interpolate_rho(rho_values_d, r, &rhoTmp, &dRho); //<---- we can change this to for speed up later.
	
		for (int k=0; k<3; k++)
                {

			 atomicAdd((float*)&(ff_d[(iOff*3) + k]), (float)(-1.0*((dfEmbed_d[iOff]+dfEmbed_d[jOff])*dRho*(dr[k]/r))));
			 //ff_d[(iOff*3) + k] = ff_d[(iOff*3) + k] - (dfEmbed_d[iOff]+dfEmbed_d[jOff])*dRho*(dr[k]/r);
			 atomicAdd((float*)&(ff_d[(jOff*3) + k]), (float)((dfEmbed_d[iOff]+dfEmbed_d[jOff])*dRho*(dr[k]/r)));
			 //ff_d[(jOff*3) + k] = ff_d[(jOff*3) + k] + (dfEmbed_d[iOff]+dfEmbed_d[jOff])*dRho*(dr[k]/r);
		}	

	    }
	}
    }


/*
    int nbrBoxes[27];

    int tx = threadIdx.x;
    int iBox = blockIdx.x;

    __shared__ int nIBox;
    __shared__ int nNbrBoxes;


    if (tx == 0)
    {
        //printf("kernel lunching!!!!!! \n");

        //number of atoms in this box
        nIBox = nA_d[iBox];

        //number of nighboring boxes
        nNbrBoxes = getNeighborBoxes(iBox, nbrBoxes);
    }

    __syncthreads();

    //algorithm

    if(tx < nIBox)
    {
         int iOff = MAXATOMS * iBox + tx;

        // loop over neighbor boxes of iBox (some may be halo boxes)
        for (int jTmp=0; jTmp<nNbrBoxes; jTmp++)
        {
            int jBox = nbrBoxes[jTmp];
            if (jBox < iBox ) continue;

            int nJBox = nA_d[jBox];  //<-------- this can be done faster!!!!!!

            // loop over atoms in jBox
            for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++)
            {
                if ( (iBox==jBox) &&(ij <= tx) ) continue;

                double r2 = 0.0;
                real3 dr;

                for (int k=0; k<3; k++)
                {
                    dr[k]= rr_d[iOff][k] - rr_d[jOff][k];  //<--------- this can be done faster!!!!
                    r2+=dr[k]*dr[k];
                }

                if(r2>Cutoff_c[0]) continue;

                double r = sqrt(r2);


                real_t rhoTmp, dRho;
                interpolate_rho(rho_values_d, r, &rhoTmp, &dRho); //<---- we can change this to for speed up later.

                 for (int k=0; k<3; k++)
                {
                    atomicAdd((float*)&(ff_d[iOff][k]), (float)(-1*( (dfEmbed_d[iOff] + dfEmbed_d[jOff])*dRho*dr[k]/r) ));  // Changed from atomicSub to Add<---possible point for speed up but alot of conciderations to plan!!
                    atomicAdd((float*)&(ff_d[jOff][k]), (float)((dfEmbed_d[iOff] + dfEmbed_d[jOff])*dRho*dr[k]/r));

                }
            }
        }
    }

   if (tx == 0)
   {
        //printf("Kernel Ending without craching!!!!! \n");    
   }
*/
}




//eam kernel
__global__ void eamForce(float *rhobar_d, float *dfEmbed_d, float *phi_values_d, float *rho_values_d, float *f_values_d, 
                         int *nA_d, float *rr_d, float *Uu_d, float *ff_d, float *test, float *etot_d)
{

     
    __shared__ int nbrBoxes[27];

    int tx = threadIdx.x;
    int iBox = blockIdx.x;
    __shared__ float localEtot[MAXATOMS];

    __shared__ int nIBox;
    __shared__ int nNbrBoxes;

    localEtot[tx] = 0;

    if (tx == 0)
    {
        //number of atoms in this box
        nIBox = nA_d[iBox];

        //number of nighboring boxes
        nNbrBoxes = getNeighborBoxes(iBox, nbrBoxes);

    }
     
    __syncthreads();

    //algorithm

    if(tx < nIBox)
    {
      int iOff = MAXATOMS * iBox + tx;

      float rr_dloc[3];
      rr_dloc[0] = rr_d[(iOff*3)];
      rr_dloc[1] = rr_d[(iOff*3) + 1];
      rr_dloc[2] = rr_d[(iOff*3) + 2];
     

        // loop over neighbor boxes of iBox (some may be halo boxes)
        for (int jTmp=0; jTmp<nNbrBoxes; jTmp++)
        {
            int jBox = nbrBoxes[jTmp];
            if (jBox < iBox ) continue;
            

            int nJBox = nA_d[jBox];  //<-------- this can be done faster!!!!!!

            // loop over atoms in jBox
            for (int jOff=MAXATOMS*jBox,ij=0; ij<nJBox; ij++,jOff++)
            {
                if ( (iBox==jBox) &&(ij <= tx) ) continue;

                double r2 = 0.0;
		float dr[3];
                for (int k=0; k<3; k++)
                {
                    dr[k]= rr_dloc[k] - rr_d[(jOff*3)+k];  //rr_d[(iOff*3)+k] - rr_d[(jOff*3)+k];  //<--------- this can be done faster!!!!
	            r2+=dr[k]*dr[k];
                }

                if(r2>Cutoff_c[0]) continue;
    
                
                double r = sqrt(r2);
               

                float phiTmp, dPhi, rhoTmp, dRho;
                interpolate_phi(phi_values_d, r, &phiTmp, &dPhi); //<----the values_d are only read and not modified but right now are global 
                interpolate_rho(rho_values_d, r, &rhoTmp, &dRho); //<---- we can change this to for speed up later.

                for (int k=0; k<3; k++)
                {
                   atomicAdd((float*)&(ff_d[(iOff*3) + k]), (float)(-1.0*(dPhi*(dr[k]/r))));//<---possible point for speed up but alot of conciderations to plan!
		   //ff_d[(iOff*3) + k] = ff_d[(iOff*3) + k] - dPhi*(dr[k]/r);
                   atomicAdd((float*)&(ff_d[(jOff*3) + k]), (float)(dPhi*dr[k]/r));
		   //ff_d[(jOff*3) + k] = ff_d[(jOff*3) + k] + dPhi*(dr[k]/r);

                }
                // update energy terms
                // calculate energy contribution based on whether
                // the neighbor box is local or remote
                if (jBox < nLB_c[0])
                {
		  localEtot[tx] += phiTmp;
		  //atomicAdd((float*)etot_d, (float)phiTmp);       //<=----every device could have their own etot and then  reduce to a single value
                    //etot_d = etot_d + phiTmp;
                }
                else
                {
		  localEtot[tx] += 0.5*phiTmp;
		  //atomicAdd((float*)etot_d, 0.5*phiTmp);
		    //etot_d = etot_d +  0.5 * phiTmp;
                }

               
                atomicAdd((float*)&(Uu_d[iOff]), 0.5*phiTmp);
		//Uu_d[iOff] = Uu_d[iOff] +  0.5*phiTmp;
                atomicAdd((float*)&(Uu_d[jOff]), 0.5*phiTmp);
		//Uu_d[jOff] = Uu_d[jOff] + 0.5*phiTmp;

                //accumulate rhobar for each atom
                atomicAdd((float*)&(rhobar_d[iOff]), rhoTmp);
		//rhobar_d[iOff] = rhobar_d[iOff] + rhoTmp;
                atomicAdd((float*)&(rhobar_d[jOff]), rhoTmp);
		//rhobar_d[jOff] = rhobar_d[jOff] + rhoTmp;

            }
            
        }


        
    }


    __syncthreads();


    if(tx < nIBox)
    {
        int iOff = MAXATOMS * iBox + tx;

        float fEmbed, dfEmbed;
        interpolate_f(f_values_d, rhobar_d[iOff], &fEmbed, &dfEmbed);
        dfEmbed_d[iOff] = dfEmbed; // save derivative for halo exchange
	localEtot[tx] += fEmbed;
        //atomicAdd((float*)etot_d, (float)fEmbed);
        Uu_d[iOff] += fEmbed;


	//atomicAdd((float*)etot_d, localEtot[tx]);
    }
    
    for (int i = MAXATOMS / 2; i >= 1; i /= 2)
    {
      if (tx < i)
	localEtot[tx] += localEtot[tx+i];

      __syncthreads();
    }
    
    if (tx == 0)
      atomicAdd((float*)etot_d, localEtot[0]);
}




//kernel launcher
extern "C" void lunch_eamForce_kernel(float *rhobar, float *dfEmbed, float phi_n, float phi_x0, float phi_invDx, float *phi_values,
                                                                       float rho_n, float rho_x0, float rho_invDx, float *rho_values,
                                                                       float f_n, float f_x0, float f_invDx, float *f_values,
						                       int nLB, int nTB, int nHB, int *nA, int *gridS, float *rr, float *Uu, float *ff,
						                       float Cutoff, float *etot)
{
    cudaError_t cuda_ret;

    float *test;

    float *rhobar_d;
    float *dfEmbed_d;

//  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    cuda_ret = cudaMalloc((void**)&rhobar_d, nTB*MAXATOMS*sizeof(float)); //global
    if(cuda_ret != cudaSuccess) printf("total = %d, Unable to alloc 1 from kernel\n", nTB*MAXATOMS*sizeof(float));
 
    cuda_ret = cudaMalloc((void**)&dfEmbed_d, nTB*MAXATOMS*sizeof(float)); //global
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 2 from kernel\n");

    float *phi_values_d;
    cuda_ret = cudaMalloc((void**)&phi_values_d, ((int)phi_n+3)*sizeof(float)); //shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 3 from kernel\n");
    
    float *rho_values_d;
    cuda_ret = cudaMalloc((void**)&rho_values_d, ((int)rho_n + 3)*sizeof(float)); //shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 4 from kernel\n");

    float *f_values_d;
    cuda_ret = cudaMalloc((void**)&f_values_d, ((int)f_n + 3)*sizeof(float)); //shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 5 from kernel\n");

    int *nA_d; 
    float *rr_d; 
    float *Uu_d; 
    float *ff_d;
     
    cuda_ret = cudaMalloc((void**)&nA_d, (nLB + nHB)*sizeof(int));   	//shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 6 from kernel\n");
    cuda_ret = cudaMalloc((void**)&rr_d, ((nTB*MAXATOMS)*3)*sizeof(float));	//shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 7 from kernel\n");
    cuda_ret = cudaMalloc((void**)&ff_d, ((nTB*MAXATOMS)*3)*sizeof(float));	//global
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 8 from kernel\n");
    cuda_ret = cudaMalloc((void**)&Uu_d, nTB*MAXATOMS*sizeof(float));   //global
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 9 from kernel\n");

    float *etot_d;

    cuda_ret = cudaMalloc((void**)&etot_d, sizeof(float));


    //copying to constant memory.
    cuda_ret = cudaMemcpyToSymbol(nLB_c, &nLB, sizeof(int));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 10 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(nTB_c, &nTB, sizeof(int));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 11 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(gridS_c, gridS, 3*sizeof(int));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 12 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(Cutoff_c, &Cutoff, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 13 from kernel\n");

    cuda_ret = cudaMemcpyToSymbol(phi_n_c, &phi_n, 1*sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 14 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(phi_x0_c, &phi_x0, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 15 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(phi_invDx_c, &phi_invDx, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 16 from kernel\n");

    cuda_ret = cudaMemcpyToSymbol(rho_n_c, &rho_n, 1*sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 17 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(rho_x0_c, &rho_x0, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 18 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(rho_invDx_c, &rho_invDx, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 19 from kernel\n");

    cuda_ret = cudaMemcpyToSymbol(f_n_c, &f_n, 1*sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 20 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(f_x0_c, &f_x0, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 21 from kernel\n");
    cuda_ret = cudaMemcpyToSymbol(f_invDx_c, &f_invDx, sizeof(float));
     if(cuda_ret != cudaSuccess) printf("Unable to alloc 22 from kernel\n");


    //copying to global memory
    cudaMemset(rhobar_d, 0, nTB*MAXATOMS*sizeof(float));
    cudaMemset(dfEmbed_d, 0, nTB*MAXATOMS*sizeof(float));

    
    cuda_ret = cudaMemcpy(phi_values_d, phi_values, ((int)phi_n+3)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 1\n");
    cuda_ret = cudaMemcpy(rho_values_d, rho_values, ((int)rho_n+3)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 2\n");
    cuda_ret =cudaMemcpy(f_values_d, f_values, ((int)f_n+3)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 3\n");
    



    cuda_ret = cudaMemcpy(nA_d, nA, (nLB + nHB)*sizeof(int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 4\n");
    cuda_ret = cudaMemcpy(rr_d, rr, ((nTB*MAXATOMS)*3)*sizeof(float), cudaMemcpyHostToDevice); //times 3!!!!!!??????
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 5\n");

    cudaMemset(ff_d, 0, ((nTB*MAXATOMS)*3)*sizeof(float));
    cudaMemset(Uu_d, 0, nTB*MAXATOMS*sizeof(float));



    cudaMemset(etot_d, 0, sizeof(float));

    dim3 Dimgrid(nLB,1,1);
    dim3 Dimblock(MAXATOMS,1,1); //MAXATOMS = 64 

    cudaMalloc((void**)&test, 3*sizeof(float));
    cudaMemset(test, 0, 3*sizeof(float));
    
    eamForce<<<Dimgrid, Dimblock>>>(rhobar_d, dfEmbed_d, phi_values_d, rho_values_d, f_values_d, nA_d, rr_d, Uu_d, ff_d, test, etot_d);
    

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) printf("Unable to launch kernel\n");

    
    cudaMemcpy(rhobar, rhobar_d, nTB*MAXATOMS*sizeof(float), cudaMemcpyDeviceToHost);
   
    cudaMemcpy(dfEmbed, dfEmbed_d, nTB*MAXATOMS*sizeof(float), cudaMemcpyDeviceToHost);
    

    cudaMemcpy(ff, ff_d, ((nTB*MAXATOMS)*3)*sizeof(float), cudaMemcpyDeviceToHost);

 
    cudaMemcpy(Uu, Uu_d, nTB*MAXATOMS*sizeof(float), cudaMemcpyDeviceToHost);
  
    cuda_ret = cudaMemcpy(etot, etot_d, sizeof(float), cudaMemcpyDeviceToHost);

    if(cuda_ret != cudaSuccess) printf("Unable to copy from kernel\n");
    
    cudaFree(test);
    cudaFree(dfEmbed_d); 
    cudaFree(rhobar_d);
    cudaFree(phi_values_d);
    cudaFree(rho_values_d);
    cudaFree(f_values_d);
    cudaFree(nA_d);
    cudaFree(rr_d);
    cudaFree(ff_d);
    cudaFree(Uu_d);
    cudaFree(etot_d);

}




extern "C" void lunch_eamForce_kernel_2(float *dfEmbed, float rho_n, float rho_x0, float rho_invDx, float *rho_values,
                                    int nLB, int nTB, int nHB, int *nA, int *gridS, float *rr, float *ff, float Cutoff)
{
    cudaError_t cuda_ret;



    float *dfEmbed_d; 
    cuda_ret = cudaMalloc((void**)&dfEmbed_d, nTB*MAXATOMS*sizeof(float)); //global
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 1 from kernel 2\n");


    float *rho_values_d;
    cuda_ret = cudaMalloc((void**)&rho_values_d, ((int)rho_n + 3)*sizeof(float)); //shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 2 from kernel 2\n");

    int *nA_d;
    float *rr_d;
    float *ff_d;

    cuda_ret = cudaMalloc((void**)&nA_d, (nLB + nHB)*sizeof(int));      //shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 6 from kernel\n");
    cuda_ret = cudaMalloc((void**)&rr_d, ((nTB*MAXATOMS)*3)*sizeof(float));     //shared
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 7 from kernel\n");
    cuda_ret = cudaMalloc((void**)&ff_d, ((nTB*MAXATOMS)*3)*sizeof(float));     //global
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 8 from kernel\n");



    cuda_ret = cudaMemcpyToSymbol(rho_n_c, &rho_n, 1*sizeof(float));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 3 from kernel 2\n");

    cuda_ret = cudaMemcpyToSymbol(rho_x0_c, &rho_x0, sizeof(float));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 4 from kernel 2\n");

    cuda_ret = cudaMemcpyToSymbol(rho_invDx_c, &rho_invDx, sizeof(float));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 5 from kernel 2\n");

    cuda_ret = cudaMemcpyToSymbol(nLB_c, &nLB, sizeof(int));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 10 from kernel\n");

    cuda_ret = cudaMemcpyToSymbol(nTB_c, &nTB, sizeof(int));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 11 from kernel\n");
  
    cuda_ret = cudaMemcpyToSymbol(gridS_c, gridS, 3*sizeof(int));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 12 from kernel\n");

    cuda_ret = cudaMemcpyToSymbol(Cutoff_c, &Cutoff, sizeof(float));
    if(cuda_ret != cudaSuccess) printf("Unable to alloc 13 from kernel\n");




    cuda_ret = cudaMemcpy(dfEmbed_d, dfEmbed, nTB*MAXATOMS*sizeof(float), cudaMemcpyHostToDevice);   
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 2\n");

    cuda_ret = cudaMemcpy(rho_values_d, rho_values, ((int)rho_n+3)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 2\n");
    
    cuda_ret = cudaMemcpy(nA_d, nA, (nLB + nHB)*sizeof(int), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 4\n");

    cuda_ret = cudaMemcpy(rr_d, rr, ((nTB*MAXATOMS)*3)*sizeof(float), cudaMemcpyHostToDevice); //times 3!!!!!!??????
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 5\n");

    cuda_ret = cudaMemcpy(ff_d, ff, ((nTB*MAXATOMS)*3)*sizeof(float), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) printf("Unable to copy to kernel 5\n");



    dim3 Dimgrid(nLB,1,1);
    dim3 Dimblock(MAXATOMS,1,1); //MAXATOMS = 64

    eamForce_2<<<Dimgrid, Dimblock>>>(dfEmbed_d, rho_values_d, nA_d, rr_d, ff_d);


    //cudaMemcpy(dfEmbed, dfEmbed_d, nTB*MAXATOMS*sizeof(float), cudaMemcpyDeviceToHost); // not sure if I need to do this!!!!
    //cudaMemcpy(ff, ff_d, ((nTB*MAXATOMS)*3)*sizeof(float), cudaMemcpyDeviceToHost);  //or this

    cudaFree(dfEmbed_d);
    cudaFree(rho_values_d);
    cudaFree(nA_d);
    cudaFree(rr_d);
    cudaFree(ff_d);

}



