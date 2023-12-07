#!/bin/sh

make clean
make IMM
./bin/IMM IC
cd src
python3 convertHepTh.py
python3 HepTh_graph.py