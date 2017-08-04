#include <dlfcn.h>
#include <stdio.h>

void* operator new(size_t sz) {
  printf("operator new: Bytes\n");
  void* m = malloc(sz);
  if(!m) puts("out of memory");
  return m;
}

void operator delete(void* m) {
  puts("operator delete");
  free(m);
}