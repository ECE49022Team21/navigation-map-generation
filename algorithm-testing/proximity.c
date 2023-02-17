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
float32_t calc_squared_distance(float32_t x1, float32_t y1, float32_t x2, float32_t y2) {
    float32_t dy = 12430.0 * (y1-y2) / 180.0;
    printf("%f\n", dy);
    float32_t dx = 24901.0 * (x1-x2) / 360.0 * DELTA_X_MULT;
    printf("%f\n", dx);
    return dy*dy + dx*dx;
}

uint8_t get_nearest_node(uint8_t node) {
    return 0;
}
/*
int main() {
    float x1 = -86.9128053;
    float y1 = 40.4249364;
    float x2 = -86.9132483;
    float y2 = 40.4264531;
    // distance is 172.80 m
    printf("%f %f\n", x1, y1);
    printf("%f %f\n", x2, y2);
    float dist_2 = calc_squared_distance(x1, y1, x2, y2);
    printf("%f\n", dist_2);
}
*/

