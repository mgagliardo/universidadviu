#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define SOFTENING 1e-9f
/******************************************************************************/

// random number generator

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}


/******************************************************************************/

typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int x = 0; x < n; x++) {
    data[x] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {
  #pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3/n; Fy += dy * invDist3/n; Fz += dz * invDist3/n;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  int nIters = 10;  // simulation iterations
  if (argc > 1) 
  {
  if(argc < 3)  
  {
  	printf("usage: <executable> <number of N-bodies> <iterations>\n");
  return 0;
  }
  else if (argc > 3)
  {
  	printf("usage: <executable> <number of N-bodies> <iterations>\n");
  return 0;
  }
  
  else
  {
  
  nBodies = atoi(argv[1]);
  nIters = atoi(argv[2]);
  }
}
  else 
  {
  printf("usage: <executable> <number of N-bodies> <iterations>\n");
  return 0;
	}
  register double rsc, vsc, r, v, x, y, z, sq, scale;

  const float dt = 0.01f; // time step
  

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;
/*
  randomizeBodies(buf, 6*nBodies); // Init pos / vel data
*/
//////////////////////////////////////////////////////////////////////////////////////////
/*Data Generation*/
drndset(7);
    rsc = (3 * 3.1415926535897932384626433832795) / 16;
    vsc = sqrt(1.0 / rsc);
    for (int i = 0; i < nBodies; i++) {
     // mass[i] = 1.0 / nbodies;
      r = 1.0 / sqrt(pow(drnd()*0.999, -2.0/3.0) - 1);
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = rsc * r / sqrt(sq);
      p[i].x = x * scale;
      p[i].y = y * scale;
      p[i].z = z * scale;

      do {
        x = drnd();
        y = drnd() * 0.1;
      } while (y > x*x * pow(1 - x*x, 3.5));
      v = x * sqrt(2.0 / sqrt(1 + r*r));
      do {
        x = drnd()*2.0 - 1.0;
        y = drnd()*2.0 - 1.0;
        z = drnd()*2.0 - 1.0;
        sq = x*x + y*y + z*z;
      } while (sq > 1.0);
      scale = vsc * v / sqrt(sq);
      p[i].vx = x * scale;
      p[i].vy = y * scale;
      p[i].vz = z * scale;
    }
////////////////////////////////////////////////////////////////////////////////////// 
  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }

    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);

  }
  double avgTime = totalTime / (double)(nIters-1); 


  


  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  free(buf);
  return 0;
}
