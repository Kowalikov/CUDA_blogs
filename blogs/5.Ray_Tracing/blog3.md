---
layout: default
title: [TBD]
permalink: /blog3
---

<!-- blog3 content here -->
<h1> [TBD] </h1>  

<author> <b>Autor:</b> Natan Jarzyński</author>  
<!-- Key words lineup -->
<meta keywords="GPU programming, programming, coding, CUDA toolkit, CUDA, C++, GPU, programowanie równoległe, instalacja CUDA, pierwszy program CUDA, alokacja pamięci GPU, kernel CUDA, nvcc, nvidia-smi">
<!-- display keywords interactive way -->
<p> <b>Słowa kluczowe:</b>  
  <span class="keyword" style="color:cornflowerblue">GPU programming</span>,
  <span class="keyword" style="color:cornflowerblue">programming</span>,
  <span class="keyword" style="color:cornflowerblue">coding</span>,
  <span class="keyword" style="color:cornflowerblue">CUDA toolkit</span>,
  <span class="keyword" style="color:cornflowerblue">CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">C++</span>,
  <span class="keyword" style="color:cornflowerblue">GPU</span>,
  <span class="keyword" style="color:cornflowerblue">programowanie równoległe</span>,
  <span class="keyword" style="color:cornflowerblue">instalacja CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">pierwszy program CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">alokacja pamięci GPU</span>,
  <span class="keyword" style="color:cornflowerblue">kernel CUDA</span>,
  <span class="keyword" style="color:cornflowerblue">nvcc</span>,
  <span class="keyword" style="color:cornflowerblue">nvidia-smi</span>
</p>

-----------
<h3> [TBD] </h3>

[TBD]


<h4>🧰 Krok 1: Sprawdzamy, czy CUDA widzi GPU</h4>


📌 Przypominajka, jak skompilować i uruchomić program CUDA na Linuxie (na Windowsie będzie podobnie, ale z rozszerzeniem `.exe`):
```bash
nvcc kernel.cu -o kernel && chmod u+x ./kernel && ./kernel
```

<!-- make a code snippet in cpp -->
```cpp
#include <iostream>
#include <cuda_runtime.h>

    
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess && count > 0)
        printf("CUDA device available\n");
    else
        printf("No CUDA device!\n");

    return 0;
}
```


<h4> 🔍 Podsumowanie</h4>

[TBD]

-------

<!-- Back to main page -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/">Strona główna</a></p>

<!-- Previous blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog2">Poprzedni wpis</a></p>

<!-- Next blog post -->
<p><a href="https://kowalikov.github.io/CUDA_blogs/blog4">Następny wpis</a></p>
