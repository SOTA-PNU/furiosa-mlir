#ifndef FURIOSA_CODEGEN_FOPT_H
#define FURIOSA_CODEGEN_FOPT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "furiosa/Codegen/FoptDialect.h.inc"
#include "furiosa/Codegen/FoptEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "furiosa/Codegen/FoptAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "furiosa/Codegen/FoptTypes.h.inc"

#define GET_OP_CLASSES
#include "furiosa/Codegen/FoptOps.h.inc"

namespace mlir::furiosa::fopt {

/// Function attributes that make a func.func a furiosa-opt kernel.
constexpr llvm::StringLiteral kChipAttr = "fopt.chip";   // #[device(chip = N)]
constexpr llvm::StringLiteral kPeAttr = "fopt.pe";       // #[device(pe = N)]
constexpr llvm::StringLiteral kAxesAttr = "fopt.axes";   // axes![NAME = SIZE, ...], in declaration order
constexpr llvm::StringLiteral kNameAttr = "fopt.name";   // Rust name of an argument or value
constexpr llvm::StringLiteral kInoutAttr = "fopt.inout"; // a parameter the kernel writes in place (&mut)

/// Bytes per element of a furiosa-opt scalar type.
int64_t elementBytes(Type type);

} // namespace mlir::furiosa::fopt

#endif // FURIOSA_CODEGEN_FOPT_H
