# CUDA C++ tutorials blogs series:

This repo serves as the codebase for the CUDA C++ blogs series - storage and development.

## Contents

- [Blogs](##blogs) - blogs materials
- [Routines](https://github.com/Kowalikov/CUDA_blogs/tree/main/routines) - handy, often used snippets of code 
- [Templates](https://github.com/Kowalikov/CUDA_blogs/tree/main/templates) - baselines for more complex implementations

## Blogs:

1. Allocation the array on GPU:
    - [source code](./blogs/1.Array_allocation_on_GPU/)
    - [blog](/blogs/1.Array_allocation_on_GPU/blog1.md)
    - [blog medium](https://medium.com/@njarzynski15/alokacja-tablicy-na-gpu-9299ba16fa88)


## Helpful tips:

Simplest `kernel.cu` compilation:

```
nvcc kernel.cu -o kernel && chmod u+x ./kernel &&./kernel
```

## Website setup:

It's on github pages with jekyll minimal setup. Here're the [docs](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll).
Effectively all the files with md extension can be picked up and used as subpages (blogs). The `_config.yml` file is used to set up the navigation bar and other settings. 

## File structure:

The `setup.md` file contains instructions on how to set up the environment for CUDA programming, which can be linked in the blogs for reference.
The `blogs` folder contains the markdown files for each blog post, which can be edited and updated as needed. Each blog post can include code snippets, images, and links to other resources.
The `assets` folder contains images and other static files used in the blogs.
The `routines` and `templates` folders contain code snippets and templates for CUDA programming, which can be linked in the blogs for reference.