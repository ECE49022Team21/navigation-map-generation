#!/usr/bin/env python3

import serial
from pynmeagps import NMEAReader
import folium
import time

def main():
    port = '/dev/ttyACM0'
    baud = 9600
    coords = []
    with serial.Serial(port, baud) as ser:
        nmr = NMEAReader(ser)
        while True:
            start = time.perf_counter()
            raw_data, parsed_data = nmr.read()
            print(parsed_data)
            print(raw_data)
            #continue
            if parsed_data.msgID == 'GGA':
                if parsed_data.quality >= 1:
                    break
        end = time.perf_counter()
        print(f"Time to quality 1: {end-start}")
        for i in range(100):
            raw_data, parsed_data = nmr.read()
            if parsed_data.msgID == 'GGA':
                coords.append((parsed_data.lat, parsed_data.lon))
    maps = folium.Map(location=coords[0], zoom_start=15)
    for i, d in enumerate(coords):
        maps.add_child(folium.Marker(location=d, popup=i))
    maps.show_in_browser()


if __name__ == "__main__":
    main()
