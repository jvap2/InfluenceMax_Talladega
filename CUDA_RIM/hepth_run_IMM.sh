#!/bin/sh
vers=2

make clean
make IMM
./bin/IMM HT two
cd src
python3 convertHepTh.py
python3 HepTh_graph.py $vers