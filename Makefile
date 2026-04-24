.PHONY: all run clean test

CC = gcc
CFLAGS = -O3 -march=native -funroll-loops -flto
LDFLAGS = -luring

TARGET = build/hash_join
SRC = hash_join.c

all: $(TARGET)

run: $(TARGET)
	./$(TARGET) $(ARGS)

$(TARGET): $(SRC) | build
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

build:
	mkdir -p build

clean:
	rm -rf build

test: $(TARGET)
	python3 test.py