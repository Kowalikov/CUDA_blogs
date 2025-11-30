# !/bin/bash

## Setup
script_name="kernel.cu"
if [ "$1" != "" ]; then
    script_name="$1"
fi

output_name="${script_name%.*}"

## Compile and run
echo Compiling "$script_name" file with NVCC ...
nvcc "$script_name" -o ./bin/"${output_name}" -std=c++14

echo Running the compiled file ...
chmod u+x ./bin/"${output_name}" 
./bin/"${output_name}"