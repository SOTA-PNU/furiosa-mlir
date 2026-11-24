// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma %s | FileCheck %s --implicit-check-not='index <"KF{4096} / 32">'

// furiosa-opt gives a kernel 128 dynamic tensor addresses, and each K tile the plan unrolls reads a tile of each
// contraction operand at an address of its own. Llama-3.1-8B's LM head (N 128256 = 8 x 96 x 167, K 4096) fits 96
// lane groups per TRF lane only at K tiles of 32: 128 K tiles, 254 addresses, which furiosa-opt rejects
// ("CTensorAddr::Var() count must be <= TENSOR_DYNAMIC_INDICIES_MAX"). The plan keeps K tiles x operands within
// half of it (Plan.h), so the head takes whole-K column tiles: one device loop over 63 tiles of 2048 columns.

// CHECK-LABEL: func.func @head
// CHECK: "fopt.for"() <{count = 63 : i64}>

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module attributes {furiosa.launch = "{\22grid\22: [1, 2004, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22activation\22, \22dtype\22: \22bf16\22, \22shape\22: [1, 4096]}, {\22arg\22: 1, \22kind\22: \22weight\22, \22dtype\22: \22bf16\22, \22shape\22: [128256, 4096]}, {\22arg\22: 2, \22kind\22: \22output\22, \22dtype\22: \22bf16\22, \22shape\22: [1, 128256]}], \22scalars\22: []}"} {
  func.func @head(%arg0: tensor<1x4096xbf16>, %arg1: tensor<128256x4096xbf16>, %arg2: tensor<1x128256xf32>) -> tensor<1x128256xf32> attributes {furiosa.written = [2]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x128256xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x128256xf32>) -> tensor<1x128256xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x4096xbf16>, tensor<128256x4096xbf16>) outs(%1 : tensor<1x128256xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %3 = arith.extf %in : bf16 to f32
      %4 = arith.extf %in_0 : bf16 to f32
      %5 = arith.mulf %3, %4 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<1x128256xf32>
    return %2 : tensor<1x128256xf32>
  }
}
