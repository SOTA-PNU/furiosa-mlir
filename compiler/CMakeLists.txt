# The Furiosa backend, built into libtriton as an external plugin: Triton's CMake adds this directory
# when TRITON_PLUGIN_DIRS names it (see compiler/README.md).
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonFuriosa
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_furiosa.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/passes.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/runtime.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/codegen.cc
    LINK_LIBS
      FuriosaTransforms
      FuriosaRuntime
      FuriosaCodegen
  )
  target_link_libraries(TritonFuriosa PRIVATE Python3::Module nanobind-static)
endif()
