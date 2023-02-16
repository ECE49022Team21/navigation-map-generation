#include "dijkstra.h"
#include "landmarks.h"
#include <stdio.h>

uint8_t path[MAX_PATH_LENGTH];

// Min Heap using array implementation of binary tree
int is_empty(int size) {
    if (size >= 0) {
        return 0;
    } else {
        return 1;
    }
}

int parent(int heap_index) {
    return (heap_index - 1) / 2;
}

int left_child(int heap_index) {
    return 2*heap_index + 1;
}

int right_child(int heap_index) {
    return 2*heap_index + 2;
}

void swap(uint8_t* heap, int first_heap_index, int second_heap_index) {
    uint8_t temp = heap[first_heap_index];
    heap[first_heap_index] = heap[second_heap_index];
    heap[second_heap_index] = temp;
}

void shift_up(uint8_t* heap, int heap_index, float32_t* dist, int* node_heap_index) {
    int parent_index = parent(heap_index);
    while (heap_index > 0 && dist[heap[parent_index]] > dist[heap[heap_index]]) {
        node_heap_index[heap[parent_index]] = heap_index;
        node_heap_index[heap[heap_index]] = parent_index;
        swap(heap, parent_index, heap_index);
        heap_index = parent_index;
        parent_index = parent(heap_index);
    }
}

void shift_down(uint8_t* heap, int size, int heap_index, float32_t* dist, int* node_heap_index) {
    int loop;
    do {
        loop = 0;
        int max_index = heap_index;
        int l = left_child(heap_index);
        int r = right_child(heap_index);
        if (r <= size && dist[heap[r]] < dist[heap[max_index]]) {
            max_index = r;
        } else if (l <= size && dist[heap[l]] < dist[heap[max_index]]) {
            max_index = l;
        } // NOTE: have to check r before l so is complete binary tree
        if (heap_index != max_index) {
            node_heap_index[heap[heap_index]] = max_index;
            node_heap_index[heap[max_index]] = heap_index;
            swap(heap, heap_index, max_index);
            heap_index = max_index;
            loop = 1;
        }
    } while (loop);
}

void insert(uint8_t* heap, int* size, uint8_t node, float32_t* dist, int* node_heap_index) {
    *size = *size + 1;
    heap[*size] = node;
    node_heap_index[node] = *size;
    shift_up(heap, *size, dist, node_heap_index);
}

uint8_t extract_min(uint8_t* heap, int* size, float32_t* dist, int* node_heap_index) {
    // WARNING: does not check if empty
    uint8_t result = heap[0];
    heap[0] = heap[*size];
    node_heap_index[result] = -1;
    node_heap_index[heap[0]] = 0;
    *size = *size - 1;
    shift_down(heap, *size, 0, dist, node_heap_index);
    return result;
}

uint8_t get_min(uint8_t* heap) {
    return heap[0];
}


void get_path(uint8_t* prev, int destination) {
    for (int i = 0; i < MAX_PATH_LENGTH; i++) {
        path[i] = -1;
    }
    int u = destination;
    // WARNING: no check if prev is defined or destination = source
    int i = MAX_PATH_LENGTH - 1;
    while (prev[u] != -1) {
        path[i] = u;
        i--;
        u = prev[u];
    }
    path[i] = u; // prev of source is -1
    if (i != 0) {
        int j;
        for (j = 0; j < MAX_PATH_LENGTH; j++) {
            path[j] = path[i];
            i++;
            if (i == MAX_PATH_LENGTH) {
                break;
            }
        }
        j++;
        for (int k = j; k < MAX_PATH_LENGTH; k++) {
            path[k] = -1;
        }
    }
    printf("Path: \n");
    for (int i = 0; i < MAX_PATH_LENGTH; i++) {
        if (path[i] == -1) {
            break;
        }
        printf("%d: %s\n", path[i], landmarks[path[i]].name);
    }
}

void print_heap(uint8_t* heap, int size) {
    int level = 1;
    int amount = 0;
    printf("HEAP: \n");
    for (int i = 0; i <= size; i++) {
        printf("%d ", heap[i]);
        amount++;
        if (amount == level) {
            printf("\n");
            level = level * 2;
            amount = 0;
        }
    }
    printf("\n");
}
void print_node_heap_index(int* heap) {
    printf("NODE_HEAP_INDEX\n");
    for (int i = 0; i < LEN_LANDMARKS; i++) {
        printf("%d ", heap[i]);
    }
    printf("\n");
}

void print_dist(float32_t* dist) {
    printf("DIST\n");
    for (int i = 0; i < LEN_LANDMARKS; i++) {
        printf("%f, ", dist[i]);
    }
    printf("\n");
}

void print_prev(uint8_t* prev) {
    printf("PREV\n");
    for (int i = 0; i < LEN_LANDMARKS; i++) {
        printf("%d, ", prev[i]);
    }
    printf("\n");
}

void dijkstra(int source, int destination) {
    // Priority Queue
    uint8_t min_heap[LEN_LANDMARKS];
    int size = -1; // WARNING: size starts at -1 so size 0 has 1 element!

    //Dijkstra data
    float32_t dist[LEN_LANDMARKS];
    uint8_t prev[LEN_LANDMARKS];

    int node_heap_index[LEN_LANDMARKS];

    for (int i = 0; i < LEN_LANDMARKS; i++) {
        dist[i] = MAX_DISTANCE;
        prev[i] = -1; // WARNING: -1 is casted to unsigned!
        insert(min_heap, &size, i, dist, node_heap_index);
    }

    dist[source] = 0; // WARNING: every time update distance, have to update heap
    shift_up(min_heap, node_heap_index[source], dist, node_heap_index);

    //print_heap(min_heap, size);
    //print_node_heap_index(node_heap_index);

    while (!(is_empty(size))) {
        uint8_t u = get_min(min_heap);
        if (u == destination) {
            break; //WARNING: does this update prev?
        }
        extract_min(min_heap, &size, dist, node_heap_index); //NOTE: ignore return
        //print_heap(min_heap, size);
        //print_node_heap_index(node_heap_index);
        for (int i = 0; i < landmarks[u].list_len; i++) {
            int v = landmarks[u].adj_list[i];
            float32_t alt = dist[u] + landmarks[u].dist_list[i];
            if (alt < dist[v]) {
                dist[v] = alt;
                prev[v] = u;
                shift_up(min_heap, node_heap_index[v], dist, node_heap_index);
            }
        }
        //print_heap(min_heap, size);
        //print_node_heap_index(node_heap_index);
        //print_prev(prev);
        //print_dist(dist);
    }
    print_dist(dist);
    print_prev(prev);
    get_path(prev, destination);
}



