#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <sys/time.h>

//Possibly make this grid and block size dynamic or user defined?
#define GRID_SIZE 30
#define BLOCK_SIZE 512

//IPs to use for functions....probably not worth making this user defined
#define IP04 0x13
#define IP08 0x11d
#define IP16 0x1100b
#define IP32 0x1400007ULL
//64 and 128 are missing the leading 1
#define IP64 0x1bULL
#define IP128 0x87ULL

//Timings
typedef struct timer
{
  struct timeval start;
  struct timeval stop;
} Timer;
void startTime(Timer *timer);
void stopTime(Timer *timer);
float elapsedTime(Timer *timer);

//Launchers
void shift_launch(unsigned w, uint64_t c, unsigned bytes, void *data);
void log_launch(unsigned w, uint64_t c, unsigned bytes, void *data);
void table_launch(unsigned w, uint64_t c, unsigned bytes, void *data);
void bytwob_launch(unsigned w, uint64_t c, unsigned bytes, void *data);
void bytwop_launch(unsigned w, uint64_t c, unsigned bytes, void *data);
void split_launch(unsigned w, uint64_t c, unsigned bytes, void *data);
void group_launch(unsigned w, uint64_t c, unsigned bytes, void *data);

//Error checking and data generation
void usage(const char *msg);
void check_answer_w08(uint64_t c, unsigned size, uint8_t *data, uint8_t *answer);
void check_answer_w16(uint64_t c, unsigned size, uint16_t *data, uint16_t *answer);
void check_answer_w32(uint64_t c, unsigned size, uint32_t *data, uint32_t *answer);
void *random_data(unsigned seed, unsigned bytes);

//multiply singles
uint8_t multiply_single_w08(uint16_t a, uint16_t b);
uint16_t multiply_single_w16(uint32_t a, uint32_t b);
uint32_t multiply_single_w32(uint64_t a, uint64_t b);
