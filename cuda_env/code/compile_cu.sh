#!/bin/bash
file=$(basename $1)
file_extension=${file##*.}

if [ "$file_extension" != "cu" ]; then
  echo "Error: your file is not cu file."
  exit 1
fi

file_name=${file%.*}

# compiling with '-arch=compute_75' produces PTX code for devices of compute capability 7.5
# compiling with '-code=sm_75' produces binary code for devices of compute capability 7.5 from PTX code
# PTX code can be 
nvcc $file -o $file_name -arch=compute_75 -code=sm_75