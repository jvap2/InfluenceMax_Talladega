#!/bin/sh

vers=2

make clean
make IMM
./bin/IMM WK two
cd src
python3 convertwiki.py
python3 wiki_graph.py $vers