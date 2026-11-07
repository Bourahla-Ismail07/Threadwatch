CC      = gcc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -fopenmp -Isdk/include
SRC     = $(wildcard sdk/src/*.c)
build/threadwatch.dll: $(SRC) sdk/include/threadwatch.h
	mkdir -p build
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) -lm
test: build/threadwatch.dll
	python -m pytest tests/python -q
