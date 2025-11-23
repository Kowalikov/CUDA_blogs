# CUDA C++ tutorials blogs series:

This repo serves as the codebase for the CUDA C++ blogs series - storage and development.

## Contents

- [Blogs](##blogs) — blogs materials
- [Routines](./routines/) - handy, often used snippets of code 
- [Templates](./templates/) - baselines for more complex implementations

## Blogs:

1. Allocation the array on GPU:
    - [source code](./blogs/1.Array_allocation_on_GPU/)
    - [blog](https://medium.com/@njarzynski15/alokacja-tablicy-na-gpu-9299ba16fa88)


## Helpful tips:

Simplest `kernel.cu` compilation:

```
nvcc kernel.cu -o kernel && chmod u+x ./kernel &&./kernel
```