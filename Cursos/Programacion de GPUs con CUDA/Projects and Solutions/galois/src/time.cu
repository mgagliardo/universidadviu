#include "gf_gpu.h"

void startTime(Timer *t)
{
  gettimeofday(&(t->start), NULL);
}

void stopTime(Timer *t)
{
  gettimeofday(&(t->stop), NULL);
}

float elapsedTime(Timer *t)
{
  return ((float) ((t->stop.tv_sec - t->start.tv_sec) +
      (t->stop.tv_usec - t->start.tv_usec)/1.0e6));
}
