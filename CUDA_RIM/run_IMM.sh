#!/bin/sh

vers=3
data_set=1
seed_size=100
ep=.5

make clean
make IMM
./bin/IMM PLN pr
cd ../ripples
conan create conan/trng
conan create conan/nvidia-cub
conan install . --build missing -o gpu=nvidia
conan build . -o gpu=nvidia
cd build/Release/tools
./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/homo_ripples.csv -p --seed-set-size $seed_size --diffusion-model IC --epsilon $ep --streaming-gpu-workers 16 -o imm_homo_IC.json
./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/homo_ripples.csv -p --seed-set-size $seed_size --diffusion-model LT --epsilon $ep --streaming-gpu-workers 16 -o imm_homo_LT.json
python3 collectdata_ic.py $data_set $vers
python3 collectdata_lt.py $data_set $vers
cd ../../../../CUDA_RIM/src
python3 check_diff.py $vers $seed_size