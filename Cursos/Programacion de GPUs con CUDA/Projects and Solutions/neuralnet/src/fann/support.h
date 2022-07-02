/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void setupProblem(unsigned int numNodes, unsigned int maxNeighborsPerNode,
  unsigned int **nodePtrs_h, unsigned int **nodeNeighbors_h,
  unsigned int **nodeVisited_h, unsigned int **nodeVisited_ref,
  unsigned int **currLevelNodes_h, unsigned int **nextLevelNodes_h,
  unsigned int **numCurrLevelNodes_h, unsigned int **numNextLevelNodes_h);
void verify(unsigned int numNodes, unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *nodeVisited_ref, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
