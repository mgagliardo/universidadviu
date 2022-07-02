#ifndef HASH_H
#define HASH_H

/**
* Contains the interface for using the simple hash functions.
* Algorithms borrowed from cse.yorku.ca.
*/

/**
* Calculates the djb2 hash.
*
* @param[in] str String to hash
* @return        The djb2 hash of input string 
*/
unsigned long int djb2Hash(unsigned char* str);

/**
* Calculates the sdbm hash.
*
* @param[in] str String to hash
* @return        The sdbm hash of input string
*/
unsigned long int sdbmHash(unsigned char* str);

/**
* Calculates the djb2 hash.
*
* @param[in] str   String to hash
* @param[in] start Offset position to begin hashing
* @return          The djb2 hash of input string[offset] 
*/
unsigned long int djb2HashOffset(char* str, int start);

/**
* Calculates the sdbm hash.
*
* @param[in] str   String to hash
* @param[in] start Offset position to being hashing
* @return          The sdbm hash of input string 
*/
unsigned long int sdbmHashOffset(char* str, int start);

#endif // HASH_H

