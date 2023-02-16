#include "stdio.h"
#include "landmarks.h"
#include "dijkstra.h"

int main() {
    //for (int i = 0; i < LEN_LANDMARKS; i++) {
    //    printf("%s\n", landmarks[i].name);
    //}
    int source = 26;
    int destination = 1;
    printf("Source: %d, %s\n", source, landmarks[source].name);
    printf("Destination: %d, %s\n", destination, landmarks[destination].name);
    dijkstra(source, destination);
}
