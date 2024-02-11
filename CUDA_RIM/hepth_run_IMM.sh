#!/bin/sh

data_set=3
vers=3
seed_size=100
ep=.5
walk=0

if [ $vers -eq 1 ]; then
    name=one
elif [ $vers -eq 2 ]; then
    name=two
elif [ $vers -eq 3 ]; then
    name=pr
elif [ $vers -eq 4 ]; then
    name=greedy
elif [ $vers -eq 5 ]; then
    name=sigmoid
elif [ $vers -eq 6 ]; then
    name=tanh
elif [ $vers -eq 7 ]; then
    name=rpr
elif [ $vers -eq 8 ]; then
    name=bfs
else
    echo "invalid option"
fi

if [ $walk -eq 0 ]; then
    dname=csr
elif [ $walk -eq 1 ]; then
    dname=csc
else
    echo "invalid option"
fi

make clean
make IMM
./bin/IMM HT $name $dname    
cd ../ripples
conan create conan/trng
conan create conan/nvidia-cub
conan install . --build missing -o gpu=nvidia
conan build . -o gpu=nvidia
cd build/Release/tools
./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/ca-HepTh_ripples.csv -p --seed-set-size $seed_size --diffusion-model IC --epsilon $ep --streaming-gpu-workers 16 -o imm_hepth_IC.json
./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/ca-HepTh_ripples.csv -p --seed-set-size $seed_size --diffusion-model LT --epsilon $ep --streaming-gpu-workers 16 -o imm_hepth_LT.json
python3 collectdata_ic.py $data_set $vers
python3 collectdata_lt.py $data_set $vers
cd ../../../../CUDA_RIM/src
python3 convertHepTh.py
python3 HepTh_graph.py $vers $seed_size $dname
cd ../../RIM_res
python3 test_communities.py $data_set $vers