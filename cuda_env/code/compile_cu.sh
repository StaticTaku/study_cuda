#!/bin/bash
file=$(basename $1)
file_extension=${file##*.}

if [ "$file_extension" != "cu" ]; then
  echo "Error: your file is not cu file."
  exit 1
fi

file_name=${file%.*}
output_extension=".out"
# compiling with '-arch=compute_75' produces PTX code for devices of compute capability 7.5
# compiling with '-code=sm_75' produces binary code for devices of compute capability 7.5 from PTX code
# PTX code can be 
if [[ "$*" == *"--openmp"* ]]; then
  nvcc -c $file -Xcompiler -fopenmp -Iinclude -arch=compute_80 -code=sm_80
  nvcc $file_name.o -o $file_name$output_extension -lgomp
  rm $file_name.o
else
  nvcc $file -o $file_name$output_extension -Iinclude -arch=compute_80 -code=sm_80 -lgomp
fi