add_triton_library(FuriosaCodegen
  Dialect.cpp
  EmitRust.cpp
  LinalgToFopt.cpp
  Mapping/Elementwise.cpp

  DEPENDS
    FoptTableGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRFuncInlinerExtension
    MLIRIR
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
    MLIRTensorDialect
)
