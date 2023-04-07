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
nvcc $file -o $file_name$output_extension -arch=compute_80 -code=sm_80