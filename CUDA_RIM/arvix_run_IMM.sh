#!/bin/sh

vers=3

make clean
make IMM
./bin/IMM AR pr
cd src
python3 convertarvix.py
python3 arvix_graph.py $vers