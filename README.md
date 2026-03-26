# CUDA C++ tutorials blogs series:

To repo służy jako baza kodu dla serii blogów o CUDA C++. Seria blogów jest dostępna na GitHub Pages pod adresem [https://kowalikov.github.io/CUDA_blogs/](https://kowalikov.github.io/CUDA_blogs/).

## Kontent:

- [Wpisy i kod źródłowy](https://github.com/Kowalikov/CUDA_blogs/tree/main/routines/blogs) - matriały do blogów, w tym kod źródłowy i markdowny
- [Rutyny](https://github.com/Kowalikov/CUDA_blogs/tree/main/routines) - Poręczne rutynowe snippedy kodu
- [Templatki](https://github.com/Kowalikov/CUDA_blogs/tree/main/templates) - baseline'y na złożone implementacje

## Wpisy:

1. [CUDA C++ Hello World: Alokacja tablicy na GPU](/blogs/1.Array_allocation_on_GPU/blog1.md)
2. [Rozeznanie środowiska CUDA - jak poznać specyfikację GPU](/blogs/2.Sanity_check/blog2.md)
3. [Pierwszy kernel CUDA](/blogs/3.Kernel_writing/blog3.md)
4. [Pierwszy benchmark GPU](/blogs/4.Benchmark_GPU/blog4.md)
5. [Prosty Ray Tracing](/blogs/5.Ray_Tracing/blog5.md)

## Linki na Medium:

1. [CUDA C++ Hello World: Alokacja tablicy na GPU](https://medium.com/@njarzynski15/alokacja-tablicy-na-gpu-9299ba16fa88)


## Poradniki:

Naprostsza kompilacja `kernel.cu` na linuxie:

```
nvcc kernel.cu -o kernel && chmod u+x ./kernel &&./kernel
```

## Setup strony z blogami:

Znajduje się ona na GitHub Pages, z minimalną konfiguracją Jekyll - [dokumentacja](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll) z instrukcją do konfiguracji. Efektywnie, wszystkie pliki z rozszerzeniem md mogą być użyte jako podstrony (blogi). Plik `_config.yml` służy do ustawienia paska nawigacyjnego i innych ustawień.

## Struktura plików:

`setup.md` zawiera instrukcje jak ustawić środowisko do programowania w CUDA.
`blogs` folder zawiera markdowny dla każdego wpisu na blogu, które można edytować i aktualizować w razie potrzeby. Każdy wpis na blogu może zawierać fragmenty kodu i linki do innych zasobów.
`assets` folder zawiera obrazy i inne statyczne pliki używane w blogach.
`routines` i `templates` foldery zawierają snippety kodu i szablony do programowania w CUDA, które można linkować w blogach jako odniesienie.
Domyślnie routines to snippety kodu, które są krótsze i bardziej ogólne, podczas gdy templates to bardziej rozbudowane implementacje, które mogą służyć jako rusztowanie i dobry punkt wyjścia do bardziej złożonych projektów.