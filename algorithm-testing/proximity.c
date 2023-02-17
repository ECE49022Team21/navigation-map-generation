#include "stdio.h"
#include "proximity.h"
#include "landmarks.h"
#include "k_d_tree.h"
#include "dijkstra.h"

static int parent(int heap_index) {
    return (heap_index - 1) / 2;
}

static int left_child(int heap_index) {
    return 2*heap_index + 1;
}

static int right_child(int heap_index) {
    return 2*heap_index + 2;
}

// Euclidean Geodesic Approximation with small distance assumption
// OR
// Projection to cartesian coordinate system and use pythogorean theorem
float32_t calc_squared_distance(float32_t x1, float32_t y1, float32_t x2, float32_t y2) {
    float32_t x_2 = (x1-x2)*(x1-x2);
    float32_t y_2 = (y1-y2)*(y1-y2);
    return x_2 + y_2;
}

uint8_t get_nearest_node(uint8_t node) {
    return 0;
}

