#include "pbf.h"

int calculatePBFSize(int k, int n, float p)
{
   double e = 0.1;
   return (unsigned int)((-k * n * p) / log(1.0 - e));
}


float calculatePBFProb(int f, int n)
{
   double e = 0.1;
   double ratio = 0.7;
   double a = (n - f) * log(1.0 - e) - (n * log(e));
   double b = (double)(n * f);
   return (float)(ratio * a / b);
}


/**
* Responsible for calculating the stats relating to a pbf.
*/
void writeStats(FILE* pbfOutput,int* actual,int* counts,int numCounts,
	int k,float p,int n,int m){
	int i  = 0;
	for(;i<numCounts;i++){
		float a = 0.0f;	
		float b = 0.0f;
		a = (float)k*p*n+m*logf(1-(float)counts[i]/k);
		b = (float)(k-m)*p;
		float f = a/b;
		a = ((float)k*p*n+m*logf(1-(float)counts[i]/k+(float)1.96*sqrt((1-(float)(k-counts[i])/k)*(k-counts[i])/k/k)));
		b = (float)(k-m)*p;
		//float fMin = a/b;
		a = ((float)k*p*n+m*logf(1-(float)counts[i]/k-(float)1.96*sqrt((1-(float)(k-counts[i])/k)*(k-counts[i])/k/k)));
		//float fMax = a/b;
		float err = (f-actual[i])/(float)actual[i];		
		fprintf(pbfOutput,"%d %d %.2f %d %.4f\n",i,counts[i],f,actual[i],err);
	}
}

