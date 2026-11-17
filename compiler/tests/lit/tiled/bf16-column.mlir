// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s

// A bf16 tile's column holds whole 8-byte units (4 bf16) per slice: of 14 x 33, the even 14 would leave 2 bf16
// per slice, so the column is the innermost 33, padded to whole packets (40), with the 14 as rows.

// CHECK-LABEL: func.func @bf16_abs
// CHECK: !fopt.dm<bf16, <"1">, <"1 # 2">, <"A0{14} # 256">, <"CP{40}">>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @bf16_abs(%x: tensor<14x33xbf16>, %o: tensor<14x33xbf16>) -> tensor<14x33xbf16> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<14x33xbf16>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<14x33xbf16>) outs(%e : tensor<14x33xbf16>) {
    ^bb0(%a: bf16, %out: bf16):
      %y = math.absf %a : bf16
      linalg.yield %y : bf16
    } -> tensor<14x33xbf16>
    return %r : tensor<14x33xbf16>
  }
}
