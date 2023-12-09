#!/bin/sh
vers=3

make clean
make IMM
./bin/IMM HT pr
cd src
python3 convertHepTh.py
python3 HepTh_graph.py $vers