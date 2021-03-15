NAME = ans

# Release products.
DST_BIN = dist/$(NAME)
DST = $(DST_BIN)

# All source files.
SRC_ALL = $(wildcard *.cc)

# All object files to be compiled.
OBJ_ALL = $(SRC_ALL:%=build/%.o)

all: dist/main

dist/main: $(OBJ_ALL)
	clang++ -std=c++17 -O2 -o $@ $^

build/%.o: %
	clang++ -std=c++17 -g -O2 -Itb -MD -o $@ -c $<

clean:
	rm -rf build dist
	mkdir build dist

-include $(OBJ_ALL:.o=.d)
