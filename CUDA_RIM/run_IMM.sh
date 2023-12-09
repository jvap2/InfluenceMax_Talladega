#!/bin/sh

vers=3

make clean
make IMM
./bin/IMM PLN pr
cd src
python3 check_diff.py $vers