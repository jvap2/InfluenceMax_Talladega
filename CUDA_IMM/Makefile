PLATFORM = $(shell uname -m)
CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
NVCC := nvcc

NVCC_FLAGS := -g -G -Xcompiler -Wall -rdc=true -gencode arch=compute_89,code=sm_89
LINK_FLAGS := -arch=sm_89  
INC_FLAGS := -lcudadevrt -lcurand -lcusparse -lcublas 
# C_FLAGS := -std=c++11 

INC := include
OBJ := obj
SRC := src
BIN := bin


IMM: $(BIN)/IMM

$(BIN)/IMM: $(OBJ)/main.o $(OBJ)/nodesel.o $(OBJ)/data.o $(OBJ)/sampling.o
	$(NVCC) $(LINK_FLAGS) $(INC_FLAGS) $^ -o $@


$(OBJ)/nodesel.o: $(SRC)/nodeselection.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/data.o: $(SRC)/dataprocessing.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/main.o: $(SRC)/main.cpp
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/sampling.o: $(SRC)/sampling.cu 
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@


clean:
	rm -rf $(OBJ)/*.o $(BIN)/*