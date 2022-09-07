CFLAGS += -g -Wall -O0 -lm -lpthread -Wno-maybe-uninitialized
WASM = wat2wasm

WAT_DIR := ./wasm
SRC_DIR := ./src
OBJ_DIR := ./build





all: seq_pc

seq_pc: $(SRC_DIR)/seq_pc.c $(SRC_DIR)/seq_pc.S
	$(CC) -o $(OBJ_DIR)/seq_pc $^ $(CFLAGS)


clean:
	rm build/*
