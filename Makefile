CC      = gcc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -fopenmp -Isdk/include
SRC     = $(wildcard sdk/src/*.c)
HDR     = $(wildcard sdk/include/*.h sdk/src/*.h)
build/threadwatch.dll: $(SRC) $(HDR)
	mkdir -p build
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) -lm
test: build/threadwatch.dll
	python -m pytest tests/python -q
clean:
	rm -rf build
.PHONY: test clean
