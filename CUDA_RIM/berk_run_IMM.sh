#!/bin/sh

vers=8
seed_size=350
ep=.5
data_set=9
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

# make clean
# make IMM
# ./bin/IMM BRK $name $dname
cd ../ripples
# conan create conan/trng
# conan create conan/nvidia-cub
# conan install . --build missing -o gpu=nvidia
# conan build . -o gpu=nvidia
cd build/Release/tools
./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/berk.tsv -p --seed-set-size $seed_size --diffusion-model IC --epsilon $ep --streaming-gpu-workers 8 -o imm_berk_IC.json
# ./imm -i /home/jvap2/Desktop/Code/Infl_Max/Graph_Data_Storage/berk.tsv -p --seed-set-size $seed_size --diffusion-model LT --epsilon $ep --streaming-gpu-workers 16 -o imm_berk_LT.json
# python3 collectdata_ic.py $data_set $vers
# python3 collectdata_lt.py $data_set $vers
# cd ../../../../CUDA_RIM/src
# python3 convertberk.py 
# python3 berk_graph.py $vers $seed_size $dname
# cd ../../RIM_res
# python3 test_communities.py $data_set $vers