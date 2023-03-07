#!/usr/bin/env python3
import folium
import re
from shapely.geometry import Point
from shapely.ops import transform
import pyproj


def main():
    filename = "sample_output_home.log"

    with open(filename) as f:
        lines = f.readlines()
    lines = [i.strip() for i in lines if i != '\n']
    coords = []
    for i in lines:
        if re.match(r"Latitude: [0-9]+\.[0-9]+ Longitude: -[0-9]+\.[0-9]+$", i):
            pass
        else:
            continue
        a = i.split(" ")
        if float(a[1]) == 0 or float(a[3]) == 0:
            continue
        coords.append((float(a[1]), float(a[3])))
    maps = folium.Map(location=coords[0], zoom_start=15)
    for i, d in enumerate(coords):
        if i % 10 == 0:
            maps.add_child(folium.Marker(location=d, popup=i))
    maps.show_in_browser()

    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:26913'))  # destination coordinate system

    print("=============================")
    print(f"Number of samples: {len(coords)}")
    actual = Point(-86.942438, 40.448145)
    actual = transform(project.transform, actual)
    distances = []

    for i, d in enumerate(coords):
        a = Point(d[1], d[0])
        a = transform(project.transform, a)
        distances.append(actual.distance(a))
    print(f"Average distance apart: {sum(distances)/len(distances)} meters.")

if __name__ == "__main__":
    main()
