#!/bin/sh

make clean
make IMM
./bin/IMM IC
cd src
python3 arvix_graph.py