# Lattice Theory Project

## Description
This repository contains the code for the project of Lattice Theory course. 
The assingment was composed by two parts:
1. Find the minimun element in array using the optimistic approach and the fix point approach
2. Develop some code for the resulution of assingment of varilables considering some constraints

## Files
[first_exe.cu](./first_exe.cu) contains the code for the first task.
[second_exe_cuda.cu](./second_exe_cuda.cu) and [second_exe_openMP](./second_exe_openMP) address both the second point, but slightly different: the first one exploit CUDA, while the second one OpenMP. 
I decided to include also the latter since I think it is an easy and valid alternative for the aim of the project.

[Parser.hpp](./Parser.hpp) contains the specification of the problem: number of variables, upper bounds and constraints.

[pco_3.txt](./pco_3.txt) contains a simple example of problem specification.

Each file contains comments which give exaustive explanation of the code.

---

## How to compile and execute the code
-For the first part:
```bash
 nvcc -o first_exe first_exe.cu
./first_exe
```

-For the second part:
```bash
 nvcc -o second_exe_cuda second_exe_cuda.cu
./second_exe_cuda
```
or
```bash
 g++ -o second_exe_openMP second_exe_openMP.cpp -fopenmp
./second_exe_openMP
```
## Contributors
Giorgio Bettonte
