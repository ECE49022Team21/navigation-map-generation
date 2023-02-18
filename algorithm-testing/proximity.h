#ifndef PROXIMITY_HEADER_G
#define PROXIMITY_HEADER_G
#define DELTA_X_MULT 0.7612237308
#define MAX_DISTANCE 1000000
#include "custom_typedef.h"

typedef struct {
    float_t x;
    float_t y;
} coord_t;

uint8_t get_nearest_node(coord_t* coord, float_t* closest_distance);
float_t squared_distance_to_node(coord_t* coord, uint8_t node);
#endif
