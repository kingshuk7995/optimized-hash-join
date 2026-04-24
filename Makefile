run: hash_join
	./build/hash_join $(ARGS)

hash_join: build
	gcc -O3 -march=native -funroll-loops -flto -o build/hash_join hash_join.c -luring

build:
	mkdir -p build

clean:
	rm -rf build

benchmark: hash_join
	python3 benchmark.py