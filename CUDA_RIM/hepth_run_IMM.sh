#!/bin/sh

make clean
make IMM
./bin/IMM HT
cd src
python3 convertHepTh.py
python3 HepTh_graph.py