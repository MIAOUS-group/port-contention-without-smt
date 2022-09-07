CFLAGS += -g -Wall -O0 -lm -lpthread -Wno-maybe-uninitialized
WASM = wat2wasm
# WFLAGS += -vvv

WAT_DIR := ./wat
SRC_DIR := ./native
OBJ_DIR := ./build
WAT_FILES := $(wildcard $(WAT_DIR)/*.wat)
OBJ_FILES := $(patsubst $(WAT_DIR)/%.wat,$(OBJ_DIR)/%.wasm,$(WAT_FILES))

all: $(OBJ_DIR) $(OBJ_FILES)

# seq-pc:
# 	$(WASM) $(WAT_DIR)/seq-pc.wat -o $(OBJ_DIR)/seq-pc.wasm $(WFLAGS)


$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(OBJ_DIR)/%.wasm: $(WAT_DIR)/%.wat
	$(WASM) -o $@ $< --enable-threads



seq-pc:
clean:
	rm build/*
