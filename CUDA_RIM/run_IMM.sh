#!/bin/sh

vers=2

make clean
make IMM
./bin/IMM PLN two
cd src
python3 check_diff.py $vers