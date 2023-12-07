#!/bin/sh

make clean
make IMM
./bin/IMM IC
cd src
python3 check_diff.py