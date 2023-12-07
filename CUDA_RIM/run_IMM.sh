#!/bin/sh

make clean
make PLN
./bin/IMM IC
cd src
python3 check_diff.py