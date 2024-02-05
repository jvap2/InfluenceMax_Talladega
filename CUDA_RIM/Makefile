PLATFORM = $(shell uname -m)
CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
NVCC := nvcc

NVCC_FLAGS := --extended-lambda -Xcompiler -O3 -rdc=true -gencode arch=compute_89,code=sm_89
# NVCC_FLAGS := --extended-lambda -g -G -Xcompiler -Wall -rdc=true -gencode arch=compute_89,code=sm_89
LINK_FLAGS := -arch=sm_89  
INC_FLAGS := -lcudadevrt -lcurand -lcusparse -lcublas -lnvToolsExt -lcusolver
# C_FLAGS := -std=c++11 

INC := include
OBJ := obj
SRC := src
BIN := bin


IMM: $(BIN)/IMM

$(BIN)/IMM: $(OBJ)/main.o $(OBJ)/rim.o $(OBJ)/data.o $(OBJ)/err.o $(OBJ)/pr.o $(OBJ)/bfs.o
	$(NVCC) $(LINK_FLAGS) $(INC_FLAGS) $^ -o $@



$(OBJ)/data.o: $(SRC)/dataprocessing.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/main.o: $(SRC)/main.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/rim.o: $(SRC)/rim.cu 
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/pr.o: $(SRC)/pr.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/err.o: $(INC)/GPUErrors.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@

$(OBJ)/bfs.o: $(SRC)/prob_bfs.cu
	$(NVCC) $(NVCC_FLAGS) $(INC_FLAGS) -c $< -o $@


clean:
	rm -rf $(OBJ)/*.o $(BIN)/*