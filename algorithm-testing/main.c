#include "stdio.h"
#include "landmarks.h"
#include "dijkstra.h"
#include "proximity.h"

void single_run(coord_t* coord) {
    float_t closest_distance;

    uint8_t nearest_node = get_nearest_node(coord, &closest_distance);
    printf("Distance to nearest node: %f\n", closest_distance);
    int source = nearest_node;
    int destination = 1;
    printf("Source: %d, %s\n", source, landmarks[source].name);
    printf("Destination: %d, %s\n", destination, landmarks[destination].name);
    dijkstra(source, destination);
}

int main() {
    coord_t coord;
    // on oval near university: 40.4251986 -86.9149741: 22.69 m
    coord.x = -86.9149741;
    coord.y = 40.4251986;

    single_run(&coord);

    // between CL50 and SC: 40.4264039 -86.9147345: 27.83 m
    coord.x = -86.9147345;
    coord.y = 40.4264039;

    single_run(&coord);
}
