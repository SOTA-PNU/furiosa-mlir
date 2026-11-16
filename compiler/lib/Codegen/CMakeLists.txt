add_triton_library(FuriosaCodegen
  Dialect.cpp
  EmitRust.cpp
  Hoist.cpp
  LinalgToFopt.cpp
  Mapping/Contraction.cpp
  Mapping/Copy.cpp
  Mapping/Elementwise.cpp
  Mapping/Gather.cpp
  Mapping/Rowwise.cpp
  Tile/Analysis.cpp
  Tile/Lower.cpp
  Tile/Ops.cpp
  Steps.cpp

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
