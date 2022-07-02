
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>



/****************NTHREADS****************/

#define T1 512
#define T2 512
#define T3 128
#define T4 64
#define T5 256
#define T6 1024

/*****************BLOCK COUNT**************/

#define F1 3
#define F2 3
#define F3 6
#define F4 6
#define F5 5
#define F6 1

#define WARPSIZE 32
#define MAXDEPTH 32
__device__ int d_step,d_bottom,d_maxdepth;
__device__ unsigned int d_blkcnt;
__device__ float d_radius;


/*************INITIALIZATION***************/

__global__ void InitializationKernel()
{
  d_step = -1;
  d_maxdepth = 1;
  d_blkcnt = 0;
  //printf("DEBUG---------> InitializationKernel");
}

/****************BOUNDING BOX KERNEL************************/

__global__ void BoundingBoxKernel(int d_nnodes, int d_nbodies, volatile int * __restrict d_start, 
			volatile int * __restrict d_child, volatile float * __restrict  d_mass, volatile float * __restrict d_posx, 
			volatile float * __restrict d_posy, volatile float * __restrict d_posz, volatile float * __restrict d_maxx,
 			volatile float * __restrict d_maxy, volatile float * __restrict d_maxz, volatile float * __restrict d_minx, 
			volatile float * __restrict d_miny, volatile float * __restrict d_minz)
{
	register int i,j,k,l;
	 register float val,minx,maxx,miny,maxy,minz,maxz;
	__shared__ volatile float sminx[T1],sminy[T1],sminz[T1],smaxx[T1],smaxy[T1],smaxz[T1];
	
	//initial min and max for x,y,z for each thread
	
	minx = maxx = d_posx[0];
	miny = maxy = d_posy[0];
	minz = maxz = d_posz[0];
	
	//go through all the bodies
	//printf("minx,miny,minz %d ,%d, %d",minx,miny,minz);
	i=threadIdx.x;
	l=T1*gridDim.x;
	
	for (j = i + blockIdx.x * T1; j < d_nbodies; j += l)
	{
    val = d_posx[j];
    minx = fminf(minx, val);
    maxx = fmaxf(maxx, val);
    val = d_posy[j];
    miny = fminf(miny, val);
    maxy = fmaxf(maxy, val);
    val = d_posz[j];
    minz = fminf(minz, val);
    maxz = fmaxf(maxz, val);
  	}
	
	//reduction of each thread in shared memory
	
	sminx[i] = minx;
	smaxx[i] = maxx;
	sminy[i] = miny;
	smaxy[i] = maxy;
	sminz[i] = minz;
	smaxz[i] = maxz;
	
	//do a binary sort and find a minimum in a block
	
	for(j = T1/2; j > 0 ; j /= 2)
	{
		__syncthreads();
		if(i < j)
		{
			k = i + j;
			sminx[i] = minx = fminf(minx,sminx[k]);
			sminy[i] = miny = fminf(miny,sminy[k]);
			sminz[i] = minz = fminf(minz,sminz[k]);
			smaxx[i] = maxx = fmaxf(maxx,smaxx[k]);
			smaxy[i] = maxy = fmaxf(maxy,smaxy[k]);
			smaxz[i] = maxz = fmaxf(maxz,smaxz[k]);
		}
	}
	
	//write to global memory for each block the min and max positions
	
	if(i == 0)
	{
		
		k = blockIdx.x;
		d_minx[k] = minx;
		d_miny[k] = miny;
		d_minz[k] = minz;
		
		d_maxx[k] = maxx;
		d_maxy[k] = maxy;
		d_maxz[k] = maxz;
		
		__threadfence();
		
	l = gridDim.x - 1;
	if( l == atomicInc(&d_blkcnt,l))
	{
		for(j = 0 ;j<= l; j++)
		{
			minx = fminf(minx,d_minx[j]);
			miny = fminf(miny,d_miny[j]);
			minz = fminf(minz,d_minz[j]);
			maxx = fmaxf(maxx,d_maxx[j]);
			maxy = fmaxf(maxy,d_maxy[j]);
			maxz = fmaxf(maxz,d_maxz[j]);
			
		}
		
	val = fmaxf (maxx - minx, maxy - miny);
	d_radius = fmaxf(val,maxz-minz) * 0.5f;
	
	k=d_nnodes;
	d_bottom = k;
	
	d_mass[k] = -1.0f;
	d_start[k] = 0;
	
	d_posx[k] = (minx + maxx) * 0.5f;
	d_posy[k] = (miny + maxy) * 0.5f;
	d_posz[k] = (minz + maxz) * 0.5f;
	
	k *= 8;
	for(i = 0; i < 8; i++)
	d_child[k+i] = -1;
	
	d_step = d_step + 1;
	}
	}
	}
	
/************END OF BOUNDING BOX KERNEL*******************/	
__global__ void InitializationKernel1(int d_nodes,int d_nbodies,volatile int * __restrict d_child)
{
	//Initialize for Tree Building Kernel
	register int k,l,top,bottom;
	top = 8 * d_nodes;
	bottom = 8 * d_nbodies;
	
	l = blockDim.x *gridDim.x;
	//align threads to warpsize for memory coalescing
	k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;
	if(k < bottom)
	k += l;
	while(k<top)
	{
		d_child[k] = -1;
		k+=l;
	}
}


/***********************Building the Tree kernel******************************************/

__global__ void TreeBuildingKernel(int nnodesd, int nbodiesd, volatile int * __restrict childd, volatile float * __restrict posxd, volatile float * __restrict posyd, volatile float * __restrict poszd)
{
  register int i, j, depth, localmaxdepth, exit, l;
  register float x, y, z, r;
  register float px, py, pz;
  register float dx, dy, dz;
  register int ch, n, cell, locked, p;
  register float radius, rootx, rooty, rootz;

  // cache root data
  radius = d_radius;
  rootx = posxd[nnodesd];
  rooty = posyd[nnodesd];
  rootz = poszd[nnodesd];

  localmaxdepth = 1;
  exit = 1;
  l = blockDim.x * gridDim.x;
  i = threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all bodies assigned to thread
  while (i < nbodiesd) {
  //printf("DEBUG---------------> TREE BUILDING KERNEL - 1, i am here 1" );
    if (exit != 0) {
      // new body, so start traversing at root
      exit = 0;
      px = posxd[i];
      py = posyd[i];
      pz = poszd[i];
      n = nnodesd;
      depth = 1;
      r = radius * 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
      if (rootx < px) {j = 1; dx = r;}
      if (rooty < py) {j |= 2; dy = r;}
      if (rootz < pz) {j |= 4; dz = r;}
      x = rootx + dx;
      y = rooty + dy;
      z = rootz + dz;
    }
	//printf("DEBUG---------------> TREE BUILDING KERNEL - 2" );
	//printf("x,y,z %d, %d, %d",x,y,z);
 // follow path to leaf cell
    ch = childd[n*8+j];
    while (ch >= nbodiesd) {
      n = ch;
      depth++;
      r *= 0.5f;
      dx = dy = dz = -r;
      j = 0;
      // determine which child to follow
	  //if not null create space
      if (x < px) {j = 1; dx = r;}
      if (y < py) {j |= 2; dy = r;}
      if (z < pz) {j |= 4; dz = r;}
      x += dx;
      y += dy;
      z += dz;
      ch = childd[n*8+j];
    }

    if (ch != -2) {  // skip if child pointer is locked and try again later
      locked = n*8+j;
      if (ch == -1) {
        if (-1 == atomicCAS((int *)&childd[locked], -1, i)) {  // if null, just insert the new body
          localmaxdepth = max(depth, localmaxdepth);
          i += l;  // move on to next body
          exit = 1;
        }
      } else {  // there already is a body in this position
        if (ch == atomicCAS((int *)&childd[locked], ch, -2)) {  // try to lock
          p = -1;
          // create new cell(s) and insert the old and new body
          do {
            depth++;

            cell = atomicSub((int *)&d_bottom, 1) - 1;
            
		//printf("DEBUG---------------> TREE BUILDING KERNEL - 3, i am here 1" );
            if (p != -1) {
              childd[n*8+j] = cell;
            }
            p = max(p, cell);

            j = 0;
            if (x < posxd[ch]) j = 1;
            if (y < posyd[ch]) j |= 2;
            if (z < poszd[ch]) j |= 4;
            childd[cell*8+j] = ch;

            n = cell;
            r *= 0.5f;
            dx = dy = dz = -r;
            j = 0;
            if (x < px) {j = 1; dx = r;}
            if (y < py) {j |= 2; dy = r;}
            if (z < pz) {j |= 4; dz = r;}
            x += dx;
            y += dy;
            z += dz;

            ch = childd[n*8+j];
            // repeat until the two bodies are different children
          } while (ch >= 0);
          childd[n*8+j] = i;

          localmaxdepth = max(depth, localmaxdepth);
          i += l;  // move on to next body
          exit = 2;
        }
      }
    }
    __syncthreads();  // __threadfence();

    if (exit == 2) {
      childd[locked] = p;
    }
  }
  // record maximum tree depth
  atomicMax((int *)&d_maxdepth, localmaxdepth);
}


__global__ void InitializationKernel2(int nnodesd, volatile int * __restrict startd, volatile float * __restrict massd)
{
  int k, l, bottom;

  bottom = d_bottom;
  l = blockDim.x * gridDim.x;
  k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x * blockDim.x;  // align to warp size
  if (k < bottom) k += l;

  // iterate over all cells assigned to thread
  while (k < nnodesd) {
    massd[k] = -1.0f;
    startd[k] = -1;
    k += l;
  }
}


/**************************Compute Centre of gravity**************************************/

__global__ void CoGKernel(const int d_nnodes,const int d_nbodies,volatile int * __restrict d_count, 
							const int * __restrict d_child,volatile float * __restrict d_mass,volatile float * __restrict d_posx,
							volatile float * __restrict d_posy,volatile float * __restrict d_posz)
							
{
	int i,j,k,ch,l,count,bottom,flag,restart;
	float m,cm,px,py,pz;
	
	__shared__ int child[T3*8];
	__shared__ float mass[T3*8];
	
	bottom = d_bottom;
	l = blockDim.x * gridDim.x;
	
	//align threads to warpsize
	
	k = (bottom & (-WARPSIZE)) + threadIdx.x + blockIdx.x*blockDim.x;
	int tid = threadIdx.x;
	if(k <bottom)
	{
		k = k+l;
	}
	 
	restart = k;
	
	//caching the children into shared memory
	
	//keep trying for 5 times without waiting
	
	for(j=0;j<5;j++)
	{
		while(k < d_nnodes)
		{
			//Allocation order used ensures that cell's children have a lower array indices than the cell
			//therefore it is not possible that a thread will ever attempt to process a cell before it has
			//processed its children
			
			//this prevents deadlock
			if(d_mass[k] < 0.0f)
			{
				//iterate for all children
				//k represents leaf node
				for(i=0;i<8;i++)
				{
				
				/*Because the majority of cells in the octree have only bodies as children, the corresponding
				threads can immediately compute the cell data.	*/
					ch = d_child[k*8+i];
					/****Write into the cache****/
					
					//cache all the child pointers of the current cell in shared memory that point to not ready
					//children. This way the thread polls only missing children which reduces global accesses.
					
					child[i*T3 + tid] = ch;
					if(ch >= d_nbodies && ((mass[i*T3+tid] = d_mass[ch]) <0.0f))
					{
						break;
					}
				}
				
				//all children are ready
				if(i==8)
				{
					cm = 0.0f;
					px = 0.0f;
					py = 0.0f;
					pz = 0.0f;
					count = 0;
					
					//scan all threads
				for(i=0;i<8;i++)
				{
					ch = child[i*T3 + tid];
					if(ch >= 0)
					{	
						//count the bodies in all subtrees and store this information in the cells
						if(ch >= d_nbodies)
						{
						m = mass[i*T3+tid];
						count += d_count[ch];
						}
						else
						{
							m = d_mass[ch];
							count++;
						}
						
						//add child's contribution
						//printf("%d",ch);
						cm = cm + m;
						px = px + d_posx[ch] * m;
						py = py + d_posy[ch] * m;
						pz = pz + d_posz[ch] * m;
					}
				}
				//printf("%d",px);
				d_count[k] = count;
				m = 1.0f/cm;
				d_posx[k] = px *m;
				d_posy[k] = py *m;
				d_posz[k] = pz *m;
				
				__threadfence();   //to ensure data is visible before setting mass
				
				d_mass[k] = cm;
				}
				
			}
			k += l;
		
		}
		k = restart;
		
	}
	flag = 0;
	/*extra operation :move all non-null child pointers to the front */
	j=0;
	//iterate over all the cells assigned to thread
	while( k < d_nnodes)
	{
		if(d_mass[k] >= 0.0f)
		{
			k+=l;
		}
		else
		{
			if(j==0)
			{
				j = 8;
				for(i=0;i<8;i++)
				{
					ch = d_child[k*8+i];
					child[i*T3 + tid] = ch;
					if((ch < d_nbodies) || ((mass[i*T3 + tid]=d_mass[ch]) >= 0.0f))
					{
						j--;
					}
				}	
			}
			else
			{
				j=8;
				for(i=0;i<8;i++)
				{
					ch = child[i*T3 + tid];
					if((ch < d_nbodies) || (mass[i*T3 + tid]=d_mass[ch] >= 0.0f) || ((mass[i*T3 + tid]=d_mass[ch]) >= 0.0f) )
					{
						j--;
					}
				}
			}
			
			if(j==0)
			{
				cm = 0.0f;
				px = 0.0f;
				py = 0.0f;
				pz = 0.0f;
				count = 0;
				
				for(i=0;i<8;i++)
				{
					ch = child[i*T3 + tid];
					if(ch >= 0)
					{	
						
						if(ch >= d_nbodies)
						{
						m = mass[i*T3+tid];
						count += d_count[ch];
						}
						else
						{
							m = d_mass[ch];
							count++;
						}
						
						//add child's contribution
						
						cm = cm + m;
						px = px + d_posx[ch] * m;
						py = py + d_posy[ch] * m;
						pz = pz + d_posz[ch] * m;
					}
				}
				d_count[k] = count;
				m = 1.0f/cm;
				d_posx[k] = px *m;
				d_posy[k] = py *m;
				d_posz[k] = pz *m;
				
				flag = 1;
			}
			
		}
		__syncthreads();
		if(flag!=0)
		{
			d_mass[k] = cm;
			k+=l;
			flag=0;
		}
	}
	
}

/**********************END OF CoGKernel****************************/
	

/**********************SORT KERNEL**********************************/

__global__ void SortKernel(int d_nnodes, int d_nbodies, int * d_sort, int * d_count, volatile int * d_start, int * d_child)
{
	int i, j, k, ch, l, start, b;

  b = d_bottom;
  l = blockDim.x * gridDim.x;
  k = d_nnodes + 1 - l + threadIdx.x + blockIdx.x * blockDim.x;

  // iterate over all cells assigned to thread
  while (k >= b) {
    start = d_start[k];
    if (start >= 0) {
      j = 0;
      for (i = 0; i < 8; i++) {
        ch = d_child[k*8+i];
        if (ch >= 0) {
          if (i != j) {
            // move children to front 
            d_child[k*8+i] = -1;
            d_child[k*8+j] = ch;
          }
          j++;
          if (ch >= d_nbodies) {
            // child is a cell
            d_start[ch] = start;  // set start ID of child
            start += d_count[ch];  // add #bodies in subtree
          } else {
            // child is a body
            d_sort[start] = ch;  // record body in 'sorted' array
            start++;
          }
        }
      }
      k -= l;  // move on to next cell
    }
  }
}


/************************END OF SORT KERNEL*************************/



/************************Force Calculation kernel*********************/


/************************END OF FORCE CALCULATION KERNEL**********************/

__global__ void ForceKernel(int d_nnodes, int d_nbodies, float dt, float d_itsq, float epsilon,volatile int* __restrict d_sort,volatile int* __restrict d_child,volatile float* __restrict d_mass,volatile float* __restrict d_posx,volatile float* __restrict d_posy,volatile float* __restrict d_posz,volatile float* __restrict d_velx,volatile float* __restrict d_vely,volatile float* __restrict d_velz,volatile float* __restrict d_accx,volatile float* __restrict d_accy,volatile float* __restrict d_accz) 
{
	register int i,j,k,n,depth,base,sbase,i_delta,pd,nd;
	register float tempx, tempy, tempz, ax, ay, az, dx, dy, dz, temp;
	__shared__ volatile int pos[MAXDEPTH*T5/WARPSIZE], node[MAXDEPTH*T5/WARPSIZE];
	__shared__ float interm[MAXDEPTH*T5/WARPSIZE];
	
	if(threadIdx.x == 0)
	{
		temp = d_radius *2;
		interm[0] = temp*temp*d_itsq;
		for(i=1;i<d_maxdepth;i++)
		{
			interm[i]=interm[i-1]*0.25f;
			interm[i-1] +=epsilon;
		}
		interm[i-1] +=epsilon;
	}
	//SYNCTHREADS -----------> 12/03/2014
	__syncthreads();
	if(d_maxdepth<=MAXDEPTH)
	{
		 base = threadIdx.x / WARPSIZE;
		    sbase = base * WARPSIZE;
		    j = base * MAXDEPTH;
		    i_delta = threadIdx.x - sbase;
		    if(i_delta<MAXDEPTH)
		    {
		    	interm[i_delta+j] = interm[i_delta];
		    }
		    __syncthreads();
   		    __threadfence_block();
for (k = threadIdx.x + blockIdx.x * blockDim.x; k < d_nbodies; k += blockDim.x * gridDim.x) 	{
			i=d_sort[k];
			tempx=d_posx[i];
			tempy=d_posy[i];
			tempz=d_posz[i];
			
			ax = 0.0f;
			ay = 0.0f;
			az = 0.0f;
			depth =j;
			if(sbase == threadIdx.x)
			{
				pos[j] = 0;
        			node[j] = d_nnodes * 8;
			}
				do {
				pd = pos[depth];
				nd = node[depth];
        			while (pd < 8) {
        			n = d_child[nd + pd];  // load child pointer
          			pd++;
          			
          			
          			if (n >= 0) {
			        dx = d_posx[n] - tempx;
			        dy = d_posy[n] - tempy;
			        dz = d_posz[n] - tempz;
			        temp = dx*dx + (dy*dy + (dz*dz + epsilon));
			        	if ((n < d_nbodies) || __all(temp >= interm[depth])) { 	temp = rsqrt(temp);
			        	temp = d_mass[n]*temp*temp*temp;
			        	ax += dx * temp;
				      ay += dy * temp;
				      az += dz * temp;
				    } else {
				    	if (sbase == threadIdx.x) {
				    	pos[depth] = pd;
           			        node[depth] = nd;
				    	}
				    	depth++;
				    	pd=0;
				    	nd=n*8;
				    	}
				    	} else {
					    pd = 8;  // early out
					}
					}
					depth--;  // done with this level
				      } while (depth >= j);

				      if (d_step > 0) {
				      d_velx[i] += (ax-d_accx[i])*dt;
				      d_vely[i] += (ax-d_accy[i])*dt;
			              d_velz[i] +=(ax-d_accy[i])*dt;
			        	}
				      d_accx[i] = ax;
				      d_accy[i] = ay;
				      d_accz[i] = az;
			}
		}
	}



/*****************************UPDATE VELOCITY AND POSITION KERNEL****************/

__global__ void UpdateKernel(int d_nbodies, float d_dtime, float d_dthf, float *  d_posx, float *  d_posy,
							float *  d_posz,  float *  dvelx,  float *  dvely, float *  dvelz,
							float *  d_accx,   float *  d_accy,  float * d_accz)
{
 int i, l;
 float d_velx, d_vely, d_velz;
 float velhx, velhy, velhz;

  
  l = blockDim.x * gridDim.x;
  for (i = threadIdx.x + blockIdx.x * blockDim.x; i < d_nbodies; i += l) {
  
    d_velx = d_accx[i] * d_dthf;
    d_vely = d_accy[i] * d_dthf;
    d_velz = d_accz[i] * d_dthf;

    velhx = dvelx[i] + d_velx;
    velhy = dvely[i] + d_vely;
    velhz = dvelz[i] + d_velz;

    d_posx[i] += velhx * d_dtime;
    d_posy[i] += velhy * d_dtime;
    d_posz[i] += velhz * d_dtime;

    dvelx[i] = velhx + d_velx;
    dvely[i] = velhy + d_vely;
    dvelz[i] = velhz + d_velz;
  }
}
