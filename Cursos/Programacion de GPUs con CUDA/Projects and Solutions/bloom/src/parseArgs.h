#ifndef PARSEARGS_H
#define PARSEARGS_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "pbf.h"

/**
* Prints the help about the program.
* 
*/
void printHelp();

/**
* Determines if a particular argument was specified.
* @param argument The argument being looked for.
* @param args The arguments the user specified.
* @param argc The number of arguments the user specified.
*/
int wasArgSpecified(const char* argument,char** args,int argc);

/**
* Returns the value of the argument..
* @param argument The argument being searched for.
* @param args The argument list specified by the user.
* @param argc The number of arguments specifeid.
* @return Returns the specified value of the argument if it exists.
*/
char* getArgValue(const char* argument,char** args,int argc);

/**
* Sets the default parameters of the bloom filter (if a parameter is not specified this value will be used
* @param bloomOptions A pointer to the bloom filter being used.
*/
void setDefault(BloomOptions_t* bloomOptions);

/**
* Parse the bloom filter options specified by the user.
* @param bloomOptions The options to be created.
* @param args The arguments specified by the user.
* @param argc The number of arguments specified by the user. 
*/
void getConfiguration(BloomOptions_t* bloomOptions,char** args,int argc);

/**
* Show the detail of the bloom filter.
*/ 
void showDetails(BloomOptions_t* bloomOptions);

/**
* Responsible for writing the bloom filter to a file.
* @param *bloomOptions A pointer to the options used to describe the bloom filter.
* @param *bloom A pointer to the bloom filter. 
*/
void writeBloomFilterToFile(BloomOptions_t* bloomOptions,char* bloom);

#endif // PARSEARGS_H

