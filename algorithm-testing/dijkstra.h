#ifndef DIJKSTRA_HEADER_G
#define DIJKSTRA_HEADER_G

#include "landmarks.h"

#define MAX_DISTANCE 1000000
#define MAX_PATH_LENGTH 20
extern uint8_t path[];
void dijkstra(int, int);
#endif
