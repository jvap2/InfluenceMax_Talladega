#!/bin/sh
vers=3

make clean
make IMM
./bin/IMM HT pr
cd ../ripples
conan create conan/trng
conan create conan/nvidia-cub
conan install . --build missing -o gpu=nvidia
conan build . -o gpu=nvidia
cd build/Release/tools
./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/ca-HepTh_ripples.csv -p --seed-set-size 50 --diffusion-model IC --epsilon .5 --streaming-gpu-workers 16 -o imm_wiki.out
cd ../../../../CUDA_RIM/src
python3 convertHepTh.py
python3 HepTh_graph.py $vers