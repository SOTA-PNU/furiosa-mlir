// Python bindings of the Furiosa backend: triton._C.libtriton.furiosa. Each area binds its own
// functions in bindings/<area>.cc.

#include <nanobind/nanobind.h>

namespace nb = nanobind;

void init_furiosa_passes(nb::module_ &m);
void init_furiosa_runtime(nb::module_ &m);
void init_furiosa_codegen(nb::module_ &m);

void init_triton_furiosa(nb::module_ &m) {
  init_furiosa_passes(m);
  init_furiosa_runtime(m);
  init_furiosa_codegen(m);
}
