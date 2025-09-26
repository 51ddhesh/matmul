build_cpp: main.cpp
	g++ main.cpp -o out -std=c++20 -O3 -DLOCAL -march=native

build_cu: main.cu
	nvcc main.cu -o out -std=c++20 -O3 -DLOCAL -arch=sm_89

run:
	@./out

clean: 
	@rm -f out	
	