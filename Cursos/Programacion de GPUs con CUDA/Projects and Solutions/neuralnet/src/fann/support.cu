/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void setupProblem(unsigned int numNodes, unsigned int maxNeighborsPerNode,
  unsigned int **nodePtrs_h, unsigned int **nodeNeighbors_h,
  unsigned int **nodeVisited_h, unsigned int **nodeVisited_ref,
  unsigned int **currLevelNodes_h, unsigned int **nextLevelNodes_h,
  unsigned int **numCurrLevelNodes_h, unsigned int **numNextLevelNodes_h) {

  // Initialize node pointers
  *nodePtrs_h = (unsigned int*) malloc((numNodes + 1)*sizeof(unsigned int));
  *nodeVisited_h = (unsigned int*) malloc(numNodes*sizeof(unsigned int));
  *nodeVisited_ref = (unsigned int*) malloc(numNodes*sizeof(unsigned int));
  (*nodePtrs_h)[0] = 0;
  for(unsigned int node = 0; node < numNodes; ++node) {
    const unsigned int numNeighbors = rand()%(maxNeighborsPerNode + 1);
    (*nodePtrs_h)[node + 1] = (*nodePtrs_h)[node] + numNeighbors;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 0;
  }

  // Initialize neighbors
  const unsigned int totalNeighbors = (*nodePtrs_h)[numNodes];
  *nodeNeighbors_h = (unsigned int*) malloc(totalNeighbors*sizeof(unsigned int));
  for(unsigned int neighborIdx = 0; neighborIdx<totalNeighbors; ++neighborIdx) {
    (*nodeNeighbors_h)[neighborIdx] = rand()%numNodes;
  }

  // Initialize current level
  *numCurrLevelNodes_h = (unsigned int*) malloc(sizeof(unsigned int));
  **numCurrLevelNodes_h = numNodes/10; // Let level contain 10% of all nodes
  *currLevelNodes_h =
    (unsigned int*) malloc((**numCurrLevelNodes_h)*sizeof(unsigned int));
  for(unsigned int idx = 0; idx < **numCurrLevelNodes_h; ++idx) {
    unsigned int node;
    // Find a node that's not visited yet
    do { node = rand()%numNodes; } while((*nodeVisited_h)[node]);
    (*currLevelNodes_h)[idx] = node;
    (*nodeVisited_h)[node] = (*nodeVisited_ref)[node] = 1;
  }

  // Prepare next level containers (i.e. output variables)
  *numNextLevelNodes_h = (unsigned int*) malloc(sizeof(unsigned int));
  **numNextLevelNodes_h = 0;
  *nextLevelNodes_h = (unsigned int*) malloc((numNodes)*sizeof(unsigned int));

}

void verify(unsigned int numNodes, unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *nodeVisited_ref, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  // Initialize reference
  unsigned int numNextLevelNodes_ref = 0;
  unsigned int *nextLevelNodes_ref =
    (unsigned int*) malloc((numNodes)*sizeof(unsigned int));

  // Compute reference out
  // Loop over all nodes in the curent level
  for(unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
      ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if(!nodeVisited_ref[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited_ref[neighbor] = 1;
        nextLevelNodes_ref[numNextLevelNodes_ref] = neighbor;
        ++numNextLevelNodes_ref;
      }
    }
  }

  // Compare to reference out
  printf("\n    Comparing number of nodes generated...");
  if(numNextLevelNodes_ref != *numNextLevelNodes) {
    printf("TEST FAILED! Mismatching number of next level nodes: reference = "
      "%u, computed = %u\n\n", numNextLevelNodes_ref, *numNextLevelNodes);
    exit(0);
  } else {
    printf("TEST PASSED"
      "\n    Comparing nodes visited...");
    // Compare the visited arrays
    for(unsigned int node = 0; node < numNodes; ++node) {
      if(nodeVisited_ref[node] && !nodeVisited[node]) {
        printf("TEST FAILED! Node %u visited in reference but not in computed"
          "\n\n", node);
        exit(0);
      } else if(nodeVisited[node] && !nodeVisited_ref[node]) {
        printf("TEST FAILED! Node %u visited in computed but not in reference"
          "\n\n", node);
        exit(0);
      }
    }
    printf("TEST PASSED"
      "\n    Comparing nodes enqueued...");
    // Use to make sure each reference node is used exactly once
    unsigned int* refUsed = (unsigned int*) malloc(numNextLevelNodes_ref*sizeof(unsigned int));
    for(unsigned int refIdx = 0; refIdx < numNextLevelNodes_ref; ++refIdx) {
      refUsed[refIdx] = 0;
    }
    // For every enqueued node
    for(unsigned int idx = 0; idx < numNextLevelNodes_ref; ++idx) {
      const unsigned int node = nextLevelNodes[idx];
      // Look for the node in the reference
      unsigned int found = 0;
      for(unsigned int refIdx = 0; refIdx < numNextLevelNodes_ref; ++refIdx) {
        if(!refUsed[refIdx]) { // If we haven't used this reference node yet
          if(node == nextLevelNodes_ref[refIdx]) { // If it's a match
            // Mark it as found and used
            found = 1;
            refUsed[refIdx] = 1;
            break;
          }
        }
      }
      if(!found) {
        printf("TEST FAILED! Node %u enqueued in computed but not in reference"
          "\n\n", node, idx);
        exit(0);
      }
    }
    // Make sure all reference nodes have been used
    for(unsigned int refIdx = 0; refIdx < numNextLevelNodes_ref; ++refIdx) {
      if(!refUsed[refIdx]) {
        printf("TEST FAILED! Node %u enqueued in reference but not in computed"
          "\n\n", nextLevelNodes_ref[refIdx]);
        exit(0);
      }
    }
    printf("TEST PASSED\n\n");
  }

  free(nextLevelNodes_ref);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

