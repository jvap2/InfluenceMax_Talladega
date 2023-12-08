#!/bin/sh

make clean
make IMM
./bin/IMM PLN
cd src
python3 check_diff.py