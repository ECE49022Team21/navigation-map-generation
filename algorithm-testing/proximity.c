#include "proximity.h"
#include "dijkstra.h"
#include "k_d_tree.h"
#include "landmarks.h"
#include "stdio.h"
#include <stdint.h>

static int parent(int i) { return (i - 1) / 2; }

static int left_child(int i) { return 2 * i + 1; }

static int right_child(int i) { return 2 * i + 2; }

// Euclidean Geodesic Approximation with small distance assumption
float_t calc_squared_distance(float_t x1, float_t y1, float_t x2, float_t y2) {
  // 12430 miles / 180 degrees
  float_t dy = 69.055555555 * (y1 - y2); // miles
  // 24901 miles / 360 degrees * DELTA_X_MULT
  // DELTA_X_MULT is cos(avg(upper_lat, lower_lat))
  float_t dx = 52.65342256 * (x1 - x2); // miles
  return dy * dy + dx * dx;             // miles^2
}

void k_d_search(coord_t* coord, uint8_t root, int depth, uint8_t* closest_node, float_t* closest_distance) {
    if (root >= MAX_LEN_K_D_TREE) {
        return;
    }
    if (k_d_tree[root] >= LEN_LANDMARKS) {
        return;
    }
    int axis = depth % 2;
    int path_taken; // 0 for left, 1 for right
    if (axis == 0) {
        if (coord->x <= landmarks[k_d_tree[root]].x) {
            k_d_search(coord, left_child(root), depth+1, closest_node, closest_distance);
            path_taken = 0;
        } else {
            k_d_search(coord, right_child(root), depth+1, closest_node, closest_distance);
            path_taken = 1;
        }
    } else {
        if (coord->y <= landmarks[k_d_tree[root]].y) {
            k_d_search(coord, left_child(root), depth+1, closest_node, closest_distance);
            path_taken = 0;
        } else {
            k_d_search(coord, right_child(root), depth+1, closest_node, closest_distance);
            path_taken = 1;
        }
    }
    float_t current_distance = calc_squared_distance(coord->x, coord->y, landmarks[k_d_tree[root]].x, landmarks[k_d_tree[root]].y);
    if (current_distance < *closest_distance) {
        *closest_distance = current_distance;
        *closest_node = root;
    }
    // Start unwinding
    // If leaf node, skip
    if (left_child(root) >= MAX_LEN_K_D_TREE) {
        return;
    }
    // Distance is from point to plane
    // Note the plane is not infinite in length
    // However, it is hard to compute shortest distance to finite plane
    //  But, if k-d tree split by median of every point, can be treated as infinite plane
    if (axis == 0) {
        // Check if closest distance is greater than perpendicular distance to splitting plane
        if (*closest_distance > calc_squared_distance(coord->x, 0, landmarks[k_d_tree[root]].x, 0)) {
            if (path_taken == 0) {
                k_d_search(coord, right_child(root), depth+1, closest_node, closest_distance);
            } else {
                k_d_search(coord, left_child(root), depth+1, closest_node, closest_distance);
            }
        }
    } else {
        if (*closest_distance > calc_squared_distance(0, coord->y, 0, landmarks[k_d_tree[root]].y)) {
            if (path_taken == 0) {
                k_d_search(coord, right_child(root), depth+1, closest_node, closest_distance);
            } else {
                k_d_search(coord, left_child(root), depth+1, closest_node, closest_distance);
            }
        }
    }
}

uint8_t get_nearest_node(coord_t* coord, float_t* closest_distance) {
    *closest_distance = MAX_DISTANCE;
    uint8_t closest_node = 0;
    k_d_search(coord, 0, 0, &closest_node, closest_distance);

    return k_d_tree[closest_node];
}

float_t squared_distance_to_node(coord_t* coord, uint8_t node) {
    return calc_squared_distance(coord->x, coord->y, landmarks[node].x, landmarks[node].y);
}
