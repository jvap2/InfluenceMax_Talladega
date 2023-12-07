#!/bin/sh

make clean
make IMM
./bin/IMM IC
cd src
python3 convertarvix.py
python3 arvix_graph.py