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
import os
from shapely.geometry import Point
import binarytree

np.float = float

out_dir = "outputs"

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
    with open(f"{out_dir}/landmarks.h", "w") as f:
        f.write("#ifndef LANDMARKS_HEADER_FILE_G\n")
        f.write("#define LANDMARKS_HEADER_FILE_G\n")
        f.write(f"#define LEN_LANDMARKS {len(landmark_list)}\n")
        f.write('''
#include "custom_typedef.h"
        ''')
        f.write("""
typedef struct {
    float_t x;
    float_t y;
    char name[48];
    uint32_t list_len;
    uint8_t* adj_list;
    float_t* dist_list;
    float_t buffer_distance;
} landmark_t;
        """)
        f.write("\n\n")
        for key, value in adj_dict.items():
            to_write = f"extern uint8_t {landmarks[key]['adj_list']}[];\n"
            f.write(to_write)
        for key, value in dist_dict.items():
            to_write = f"extern float_t {landmarks[key]['dist_list']}[];\n"
            f.write(to_write)
        f.write("extern landmark_t landmarks[];\n")
        f.write("#endif //LANDMARKS_HEADER_FILE_G\n")


def write_landmarks_c(landmark_list, adj_dict, dist_dict, landmarks):
    with open(f"{out_dir}/landmarks.c", "w") as f:
        f.write('#include "landmarks.h"\n')
        for key, value in adj_dict.items():
            array_string = "{" + value.__str__()[1:-1] + "}"
            to_write = f"uint8_t {landmarks[key]['adj_list']}[] = {array_string};\n"
            f.write(to_write)
        for key, value in dist_dict.items():
            array_string = "{" + value.__str__()[1:-1] + "}"
            to_write = f"float_t {landmarks[key]['dist_list']}[] = {array_string};\n"
            f.write(to_write)
        f.write("landmark_t landmarks[] = {\n")
        for d in landmark_list:
            f.write(f"[{d['i']}] = ")
            f.write("{ ")
            struct_string = f" .x = {d['x']}, .y = {d['y']}," + \
                            f" .name = " + '"' + d['name'] + '",' + \
                            f" .list_len = {d['list_len']}, .adj_list = {d['adj_list']}, .dist_list = {d['dist_list']}," + \
                            f" .buffer_distance = {d['buffer_distance']}"
            f.write(struct_string)
            f.write(" },\n")
        f.write("};\n")


def generate_structs(G):
    landmarks = {}
    i = 0
    for node, data in G.nodes(data=True):
        x = round(data["x"], 10)
        y = round(data["y"], 10)
        buffer_distance = round(data["buffer"], 10)
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
            "dist_list": dist_list,
            "buffer_distance": buffer_distance
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
    IS_NONE = 255
    assert (len(landmark_list) - 1 < IS_NONE)
    k = []
    root = None
    depth = 0

    def sort_x(landmark_list):
        landmark_list.sort(key=lambda x: x["x"])

    def sort_y(landmark_list):
        landmark_list.sort(key=lambda x: x["y"])

    def median_split(landmark_list):
        return len(landmark_list) // 2

    """
    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:26913'))  # destination coordinate system

    for d in landmark_list:
        p = Point(d["x"], d["y"])
        new_p = transform(project.transform, p)
        d['x'] = new_p.x
        d['y'] = new_p.y
    """

    # Start with x
    def k_d_tree(landmark_list, depth):
        if len(landmark_list) == 0:
            return None
        if len(landmark_list) == 1:
            return binarytree.Node(landmark_list[0]["i"])
        root = None
        axis = depth % 2
        if axis == 0:
            sort_x(landmark_list)
        else:
            sort_y(landmark_list)

        i = median_split(landmark_list)
        root = binarytree.Node(landmark_list[i]["i"])
        root.left = k_d_tree(landmark_list[0:i], depth+1)
        root.right = k_d_tree(landmark_list[i+1:], depth+1)
        return root
    root = k_d_tree(landmark_list, depth)
    print(root.values)
    print(root)
    for i in root.values:
        if i is None:
            k.append(IS_NONE)
        else:
            k.append(i)
    return k, root


def write_k_d_tree(k):
    with open(f"{out_dir}/k_d_tree.h", "w") as f:
        f.write(
'''#ifndef K_D_TREE_HEADER
#define K_D_TREE_HEADER
#include "custom_typedef.h"
''')
        f.write(
f"""
#define MAX_LEN_K_D_TREE {len(k)}
extern uint8_t k_d_tree[];
#endif
"""
        )
    with open(f"{out_dir}/k_d_tree.c", "w") as f:
        f.write('#include "k_d_tree.h"\n')
        f.write("uint8_t k_d_tree[] = {\n")
        to_write = []
        for i in range(0, len(k), 10):
            array_string = k[i:i+10].__str__()[1:-1] + ',\n'
            to_write.append(array_string)
        to_write[-1] = to_write[-1][:-2] + '\n'
        f.writelines(to_write)
        f.write("};\n")


def write_c_array(G):
    landmark_list, adj_dict, dist_dict, landmarks = generate_structs(G)

    # Sort array again right before write
    landmark_list.sort(key=lambda x: x["i"])
    write_landmarks_header(landmark_list, adj_dict, dist_dict, landmarks)
    write_landmarks_c(landmark_list, adj_dict, dist_dict, landmarks)

    # This permanently changes landmark list so do this last
    k, root = generate_k_d_tree(landmark_list)
    write_k_d_tree(k)


def local_small():
    fp = 'purdue_map-all.osm.pbf'
    osm = pyrosm.OSM(fp)
    buildings = osm.get_buildings()
    # print(buildings)
    plot = buildings.plot(figsize=(20, 20))
    pathways = osm.get_network()
    pathways.plot(ax=plot, color="green")
    fig = plot.get_figure()
    fig.savefig(f"{out_dir}/map-all-buildings.png")


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
    fig.savefig(f"{out_dir}/simple_map.png")
    networkx.draw_networkx_edges(G, pos=pos)
    networkx.draw_networkx_edge_labels(
        G, pos=pos, edge_labels=labels, font_size=7, alpha=0.5)
    fig.savefig(f"{out_dir}/map.png")


def do_network(osm):
    nodes, edges = osm.get_network(nodes=True)
    print(nodes)
    print(edges)
    nodes.to_csv(f'{out_dir}/nodes.csv')
    edges.to_csv(f'{out_dir}/edges.csv')

    G = osm.to_graph(nodes, edges, graph_type="networkx")
    # print(G)
    # print(G.graph)
    networkx.write_adjlist(G, f"{out_dir}/adj_list.txt")
    networkx.write_multiline_adjlist(G, f"{out_dir}/adj_list_multi.txt")
    G_undirected = G.to_undirected()
    # print(G_undirected)
    networkx.write_adjlist(G_undirected, f"{out_dir}/undirected_adj_list.txt")
    networkx.write_multiline_adjlist(
        G_undirected, f"{out_dir}/undirected_adj_list_multi.txt")

    for (n1, n2, d) in G_undirected.edges(data=True):
        length = d['length']
        d.clear()
        d['length'] = length
    networkx.write_multiline_adjlist(
        G_undirected, f"{out_dir}/less_undirected_adj_list_multi.txt")


def make_pyvis(G):
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
    net.show(f"{out_dir}/pyvis-map.html")


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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
    # print(buildings.columns)
    buildings = buildings[['name', 'building', 'geometry', 'centroids']]

    pathways = osm.get_network()

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
            # print(dist)
            if dist < 100:
                G.add_edge(node, other_node, dist=dist, title=dist)

    # Plot exteriors
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    exteriors = buildings['geometry'].exterior
    print(exteriors)
    exteriors.plot(ax=ax)
    fig.savefig(f"{out_dir}/exteriors.png")

    # Find buffer distance from centroid to building exterior
    for node, data in G.nodes(data=True):
        centroid = transform(project.transform, data["centroid"])
        shape = transform(project.transform, other_data["geometry"])
        print(f"Distance from centroid to {data['name']}: {centroid.distance(shape.exterior)}")
        print(f"Hausdorff Distance from centroid to {data['name']}: {centroid.hausdorff_distance(shape.exterior)}")
        data["buffer"] = (centroid.distance(shape.exterior) + centroid.hausdorff_distance(shape.exterior)) / 2
        data["buffer"] = (data["buffer"] * 0.0006213711922)**2
    # update to epsg:26913
    for node, data in G.nodes(data=True):
        new_centroid = transform(project.transform, data["centroid"])
        data["proj_x"] = new_centroid.x
        data["proj_y"] = new_centroid.y
        data.pop("geometry")
        data.pop("centroid")
    # G = G.to_undirected()

    create_map(buildings, centroids, pathways, G)
    buildings.to_csv(f"{out_dir}/buildings.csv")
    pathways.to_csv(f'{out_dir}/pathways.csv')
    local_small()
    do_network(osm)
    # make_pyvis(G)
    write_c_array(G)

    return


if __name__ == "__main__":
    main()
