.PHONY: all run clean test benchmark

CC = gcc
CFLAGS = -O3 -march=native -funroll-loops -flto -Wall -Wextra
LDFLAGS = -luring

TARGET = build/hash_join
SRC = main.c

all: $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

$(TARGET): $(SRC) types.h mmap_io.h io_uring_io.h csv_linear_parse.h csv_simd_parse.h hash_join.h | build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build:
	mkdir -p build

clean:
	rm -rf build

test: $(TARGET)
	python3 -m venv venv && ./venv/bin/pip install pandas numpy
	./venv/bin/python test.py

benchmark: $(TARGET)
	python3 -m venv venv && ./venv/bin/pip install pandas numpy
	./venv/bin/python benchmark.py