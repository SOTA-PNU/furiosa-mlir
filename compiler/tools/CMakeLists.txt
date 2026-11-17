add_executable(furiosa-mlir-opt furiosa-mlir-opt.cpp)
target_compile_options(furiosa-mlir-opt PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
target_link_libraries(furiosa-mlir-opt PRIVATE FuriosaTransforms FuriosaCodegen TritonIR MLIROptLib MLIRPass)
