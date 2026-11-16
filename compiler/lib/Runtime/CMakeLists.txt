# L1: JIT-compiling raised linalg kernels for the host CPU.
add_triton_library(FuriosaRuntime
  Jit.cpp

  LINK_LIBS PUBLIC
    MLIRAffineDialect
    MLIRAffineToStandard
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRArithTransforms
    MLIRArithValueBoundsOpInterfaceImpl
    MLIRBufferizationDialect
    MLIRBufferizationToMemRef
    MLIRBufferizationTransforms
    MLIRBuiltinToLLVMIRTranslation
    MLIRControlFlowDialect
    MLIRControlFlowToLLVM
    MLIRControlFlowTransforms
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRIndexToLLVM
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMathDialect
    MLIRMathToLibm
    MLIRMathToLLVM
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
    MLIRParser
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRTensorDialect
    MLIRTransforms
    MLIRTensorInferTypeOpInterfaceImpl
    MLIRTensorTransforms
    MLIRUBDialect
    MLIRUBToLLVM
    MLIRValueBoundsOpInterface
    LLVMOrcJIT
    LLVMX86AsmParser
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
)

if(NOT DEFINED LLVM_LIBRARY_DIR)
  get_filename_component(LLVM_LIBRARY_DIR "${LLVM_DIR}/../.." ABSOLUTE)
endif()
target_compile_definitions(FuriosaRuntime PRIVATE
  MLIR_C_RUNNER_UTILS_PATH="${LLVM_LIBRARY_DIR}/libmlir_c_runner_utils.so")
