add_triton_library(FuriosaTransforms
  RaiseToLinalg.cpp
  Structure.cpp

  LINK_LIBS PUBLIC
    TritonIR
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRParser
    MLIRTransformUtils
    MLIRMathDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRPass
    MLIRTransforms
)
