#!/usr/bin/env python3
import re
from shapely.ops import transform
import pyproj
from functools import partial
from pyvis.network import Network
from geopy import distance
import networkx
import pyrosm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.float = float

"""
landmark struct:
landmark_t
{
    x: float32
    y: float32
    name: char[48]
    list_len: uint32_t
    adj_list: uint8_t*
    dist_list: float32*
}
"""


def write_landmarks_header(landmark_list, adj_dict, dist_dict, landmarks):
    with open("../stm32-navigation/landmarks.h", "w") as f:
        f.write("#ifndef LANDMARKS_HEADER_FILE_G\n")
        f.write("#define LANDMARKS_HEADER_FILE_G\n")
        f.write(f"#define LEN_LANDMARKS {len(landmark_list)}\n")
        f.write("""
typedef float float32_t;
typedef int uint32_t;
typedef int uint8_t;
        """)
        f.write("""
typedef struct {
    float32_t x;
    float32_t y;
    char name[48];
    uint32_t list_len;
    uint8_t* adj_list;
    float32_t* dist_list;
} landmark_t;
        """)
        f.write("\n\n")
        for key, value in adj_dict.items():
            to_write = f"extern uint8_t {landmarks[key]['adj_list']}[];\n"
            f.write(to_write)
        for key, value in dist_dict.items():
            to_write = f"extern float32_t {landmarks[key]['dist_list']}[];\n"
            f.write(to_write)
        f.write("extern landmark_t landmarks[];\n")
        f.write("#endif //LANDMARKS_HEADER_FILE_G\n")


def write_landmarks_c(landmark_list, adj_dict, dist_dict, landmarks):
    with open("../stm32-navigation/landmarks.c", "w") as f:
        f.write('#include "landmarks.h"\n')
        for key, value in adj_dict.items():
            array_string = "{" + value.__str__()[1:-1] + "}"
            to_write = f"uint8_t {landmarks[key]['adj_list']}[] = {array_string};\n"
            f.write(to_write)
        for key, value in dist_dict.items():
            array_string = "{" + value.__str__()[1:-1] + "}"
            to_write = f"float32_t {landmarks[key]['dist_list']}[] = {array_string};\n"
            f.write(to_write)
        f.write("landmark_t landmarks[] = {\n")
        for d in landmark_list:
            f.write(f"[{d['i']}] = ")
            f.write("{ ")
            struct_string = f" .x = {d['x']}, .y = {d['y']}," + \
                            f" .name = " + '"' + d['name'] + '",' + \
                            f" .list_len = {d['list_len']}, .adj_list = {d['adj_list']}, .dist_list = {d['dist_list']}"
            f.write(struct_string)
            f.write(" },\n")
        f.write("};\n")


def generate_structs(G):
    landmarks = {}
    i = 0
    for node, data in G.nodes(data=True):
        x = round(data["x"], 8)
        y = round(data["y"], 8)
        name = data["name"]
        name = re.sub("[\(\[].*?[\)\]]", "", name).lower().strip()
        name = re.sub("-|_", " ", name)
        assert (len(name) <= 47)
        list_len = 0
        adj_list = f"_Landmark_Adj_List_{node}"
        dist_list = f"_Landmark_Dist_List_{node}"
        landmarks[node] = {
            "i": i,
            "x": x,
            "y": y,
            "name": name,
            "list_len": list_len,
            "adj_list": adj_list,
            "dist_list": dist_list
        }
        i = i + 1
    adj_dict = {}
    dist_dict = {}
    for node, data in G.nodes(data=True):
        adj_list = []
        dist_list = []
        for u, v, d in G.edges(node, data=True):
            adj_list.append(landmarks[v]["i"])
            dist_list.append(round(d["dist"], 8))
        adj_dict[node] = adj_list
        dist_dict[node] = dist_list

    for key, value in landmarks.items():
        value["list_len"] = len(adj_dict[key])
        assert (len(adj_dict[key]) == len(dist_dict[key]))

    landmark_list = list(landmarks.values())
    landmark_list.sort(key=lambda x: x["i"])

    return landmark_list, adj_dict, dist_dict, landmarks


def generate_k_d_tree(landmark_list):
    k = []
    depth = 0

    def parent(i):
        return (i - 1) // 2

    def left_child(i):
        return 2*i + 1

    def right_child(i):
        return 2*i+2

    def sort_x(landmark_list):
        landmark_list.sort(key=lambda x: x["x"])

    def sort_y(landmark_list):
        landmark_list.sort(key=lambda x: x["y"])

    def median_split(landmark_list):
        return len(landmark_list) // 2

    axis = depth % 2
    if axis == 0:
        sort_x(landmark_list)
    if

    return k


def write_c_array(G):
    landmark_list, adj_dict, dist_dict, landmarks = generate_structs(G)

    # Sort array again right before write
    landmark_list.sort(key=lambda x: x["i"])
    write_landmarks_header(landmark_list, adj_dict, dist_dict, landmarks)
    write_landmarks_c(landmark_list, adj_dict, dist_dict, landmarks)


def local_small():
    fp = 'purdue_map-all.osm.pbf'
    osm = pyrosm.OSM(fp)
    buildings = osm.get_buildings()
    print(buildings)
    plot = buildings.plot()
    fig = plot.get_figure()
    fig.savefig("map.png")


def create_map(buildings, centroids, pathways, G):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    buildings.plot(ax=ax)
    centroids.plot(color="red", ax=ax)
    buildings.apply(lambda x: ax.annotate(text=x['name'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),
                    axis=1)
    # pathways.plot(ax=ax, color='green')
    pos = {}
    for node, data in G.nodes(data=True):
        pos[node] = (data["x"], data["y"])
    labels = {}
    for u, v, data in G.edges(data=True):
        labels[(u, v)] = round(data["dist"])
    networkx.draw_networkx_edges(G, pos=pos)
    networkx.draw_networkx_edge_labels(
        G, pos=pos, edge_labels=labels, font_size=7, alpha=0.5)
    fig.savefig("map.png")


def do_network(osm):
    nodes, edges = osm.get_network(nodes=True)
    print(nodes)
    print(edges)
    nodes.to_csv('nodes.csv')
    edges.to_csv('edges.csv')

    G = osm.to_graph(nodes, edges, graph_type="networkx")
    print(G)
    print(G.graph)
    networkx.write_adjlist(G, "adj_list.txt")
    networkx.write_multiline_adjlist(G, "adj_list_multi.txt")
    G_undirected = G.to_undirected()
    print(G_undirected)
    networkx.write_adjlist(G_undirected, "undirected_adj_list.txt")
    networkx.write_multiline_adjlist(
        G_undirected, "undirected_adj_list_multi.txt")

    for (n1, n2, d) in G_undirected.edges(data=True):
        length = d['length']
        d.clear()
        d['length'] = length
    networkx.write_multiline_adjlist(
        G_undirected, "less_undirected_adj_list_multi.txt")


def main():
    fp = 'purdue_map-all.osm.pbf'
    # bounding_box = [40.4241731, -86.9166476, 40.4313206, -86.9104916]
    bounding_box = [-86.9166476, 40.4241731, -86.9104916, 40.4313206]
    osm = pyrosm.OSM(fp, bounding_box=bounding_box)
    buildings = osm.get_buildings()
    buildings = buildings[buildings['name'].notna()]

    # better way to filter
    buildings = buildings[buildings['building'] != 'yes']
    buildings = buildings[buildings['building'] != 'no']

    centroids = buildings['geometry'].centroid
    buildings = buildings.assign(centroids=centroids)
    print(buildings.columns)
    buildings = buildings[['name', 'building', 'geometry', 'centroids']]

    pathways = osm.get_network()

    # buildings.to_csv("buildings.csv")
    # pathways.to_csv('pathways.csv')
    print(buildings.crs)

    G = networkx.Graph()
    for i, building in buildings.iterrows():
        d = {
            "name": building["name"],
            "x": building["centroids"].x,
            "y": building["centroids"].y,
            "geometry": building["geometry"],
            "centroid": building["centroids"],
            "label": building["name"]
        }
        G.add_node(i, **d)
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:26913'))  # destination coordinate system

    for node, data in G.nodes(data=True):
        for other_node, other_data in G.nodes(data=True):
            if node == other_node:
                continue
            # a, b = (data["y"], data["x"]), (other_data["y"], other_data["x"])
            # dist = distance.distance(a, b).m
            centroid = transform(project.transform, data["centroid"])
            shape = transform(project.transform, other_data["geometry"])
            dist = centroid.distance(shape)
            print(dist)
            if dist < 100:
                G.add_edge(node, other_node, dist=dist, title=dist)
    for node, data in G.nodes(data=True):
        data.pop("geometry")
        data.pop("centroid")
    # G = G.to_undirected()
    net = Network(height="1000px", width="1000px", directed=False)
    """
    net.set_options('''
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant":-2000,
          "centralGravity": 0.5,
          "springLength": 0,
          "springConstant": 0.015,
          "damping": 0.09,
          "avoidOverlap": 0
        },
        "maxVelocity:":50,
        "minVelocity": 0.75,
        "timestep": 0.5
      }
    }
    ''')
    """
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    # net.show("example.html")

    create_map(buildings, centroids, pathways, G)
    write_c_array(G)

    return
    osm = pyrosm.OSM(fp)
    buildings = osm.get_buildings()
    buildings = buildings[buildings['name'].notna()]

    # better way to filter
    buildings = buildings[buildings['building'] != 'yes']
    buildings = buildings[buildings['building'] != 'no']

    centroids = buildings['geometry'].centroid
    buildings = buildings.assign(centroids=centroids)
    print(buildings.columns)
    buildings = buildings[['name', 'building', 'geometry', 'centroids']]

    pathways = osm.get_network()

    buildings.to_csv("buildings_all.csv")


if __name__ == "__main__":
    main()
