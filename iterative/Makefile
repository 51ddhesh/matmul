# * -arch=sm_89 is specific to my GPU
build_cu: main.cu
	nvcc main.cu -o out -O3 -std=c++20 -DLOCAL -arch=sm_89

build_cpp: main.cpp
	g++ main.cpp -o out -O3 -std=c++20 -DLOCAL -march=native

run: 
	@./out

clean: 
	@rm -f out
