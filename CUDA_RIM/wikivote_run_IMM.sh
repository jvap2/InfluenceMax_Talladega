#!/bin/sh

make clean
make IMM
./bin/IMM IC
cd src
python3 convertwiki.py
python3 wiki_graph.py