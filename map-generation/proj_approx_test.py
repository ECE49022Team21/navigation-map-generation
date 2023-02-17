#!/usr/bin/env python3
import pyproj
from shapely.ops import transform
from shapely.geometry import Point
import matplotlib.pyplot as plt


def main():
    bounds = {
        "lower_left":  Point(-86.9166424, 40.4241218),
        "lower_right": Point(-86.9103621, 40.4240607),
        "upper_left":  Point(-86.9166942, 40.4314034),
        "upper_right": Point(-86.8946819, 40.4286344)
        }

    all_points = []
    #for i in range(bounds["lower_left"], bounds["lower_right"])


    project = pyproj.Transformer.from_proj(
        pyproj.Proj(init='epsg:4326'),  # source coordinate system
        pyproj.Proj(init='epsg:26913'))  # destination coordinate system

    transformed_bounds = {}
    for k, d in bounds.items():
        transformed_bounds[k] = transform(project.transform, d)



    point_list = [(i[0], i[1].x, i[1].y) for i in transformed_bounds.items()]
    x = [i[1] for i in point_list]
    y = [i[2] for i in point_list]
    print(point_list)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i in point_list:
        ax.annotate(i[0], (i[1], i[2]))

    fig.savefig("a.png")
    print(transformed_bounds["lower_left"])
    print(project.transform(bounds["lower_left"].x, bounds["lower_left"].y))


if __name__ == "__main__":
    main()
