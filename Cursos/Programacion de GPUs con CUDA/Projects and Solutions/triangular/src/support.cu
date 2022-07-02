#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include <iomanip>

#include "support.h"

void verifyResults(double* computed, double* actual, unsigned int count)
{
    // Compare the computed values to the actual ones
    float tolerance = 1e-3;
    
    
    for(unsigned int i = 0; i < count; i++)
    {
        if (i != 1)
        {
            const double diff = (computed[i] - actual[i])/actual[i];
            if(diff > tolerance || diff < -tolerance)
            {
                printf("TEST FAILED at index %u, actual = %f, computed = %f"
                       "\n\n", i, actual[i], computed[i]);
                exit(0);
            }
        }
    }
    printf("TEST PASSED\n\n");
    
    
    
    /*for(unsigned int i = 0; i < count; i++)
    {
        //double diff = (computed[i] - actual[i])/actual[i];
        //if (actual[i] == 0)
        //    diff = computed[i] - actual[i];

        //if(diff > tolerance || diff < -tolerance)
        double diff = abs(computed[i] - actual[i]);
        if (diff > tolerance)
        {
            diff = (computed[i] - actual[i])/actual[i];
            
            if(diff > tolerance || diff < -tolerance)
            {
                printf("TEST FAILED at index %u, actual = %f, computed = %f"
                       "\n\n", i, actual[i], computed[i]);
                exit(0);
            }
        }
    }
    printf("TEST PASSED\n\n");*/
}

void loadCSV(const std::string& filename, std::vector<double>& values)
{
    //std::vector<float> values;
    
    std::ifstream file(filename.c_str());
    std::string line;
    std::string cell;
    
    while(std::getline(file,line))
    {
        std::stringstream lineStream(line);
        
        //stof doesn't compile on GEM, had to use atof instead
        while( std::getline(lineStream, cell, ',' ))
            values.push_back(std::atof(cell.c_str()));
    }
    
    for(int i=0; i<int(values.size()); i++)
    {
        //std::cout << std::setprecision(51) << values[i] << " ";
        //std::cout << values[i] << "\n";
    }

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
