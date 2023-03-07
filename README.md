# About
Justin Lee

This folder contains subdirectories of test code, map generation, and code generation for the navigation subsystem.

This code is NOT meant for use in the final product.
However, C code generated for maps will be used.

## algorithm-testing

This folder contains test code for testing the various algorithms used in the navigation subsystem.

Run `make` to build.

## gps-point-plotting

This folder contains python code to plot recorded GPS coordinates from the GPS module on a map using folium.

## map-generation
This is python code that looks at maps from OSM and generates C code.

Note the C code generated uses a custom_type_def header file which is not generated. This is to ensure compatibility with the amd64 and arm architectures.

The python project uses a venv.
Please run 'install' to setup the venv

Run 'update_requirements' if a package is updated in the venv

Before running, make sure to source the venv

**parse_map.py**: The main python file to generate C code and maps.

The C files generated describe a map of the following area:
    south of Northwestern and Stadium Avenue,
    north of State Street,
    east of University Street,
    west of Grant Street.
Additionally, this map will not contain the new names for the Engineering and Polytechnic Gateway building (Lambertus Hall and Dudley).


## python-gps

This folder contains C code to easily read and parse NMEA messages from the GPS module using a serial connection. This is for testing purposes only.

