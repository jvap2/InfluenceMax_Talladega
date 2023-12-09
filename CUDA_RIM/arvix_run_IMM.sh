#!/bin/sh

vers=2

make clean
make IMM
./bin/IMM AR two
cd src
python3 convertarvix.py
python3 arvix_graph.py $vers