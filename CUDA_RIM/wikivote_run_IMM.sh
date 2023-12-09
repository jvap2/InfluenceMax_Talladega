#!/bin/sh

vers=3

make clean
make IMM
./bin/IMM WK pr
cd src
python3 convertwiki.py
python3 wiki_graph.py $vers