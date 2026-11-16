add_triton_library(FuriosaTransforms
  RaiseToLinalg.cpp

  LINK_LIBS PUBLIC
    TritonIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRPass
    MLIRTransforms
)
