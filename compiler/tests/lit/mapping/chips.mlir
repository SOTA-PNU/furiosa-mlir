// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma --furiosa-ve-fuse %s | FileCheck %s

// SPMD over two chips (furiosa.chips = 2): the kernel runs on both chips (#[device(chip = 2)]), each on its own copy
// of the buffers: every chip mapping is the CHIP axis, which the kernel declares.

// CHECK-LABEL: func.func @accumulate
// CHECK-SAME: (%arg0: !fopt.hbm<f32, <"CHIP{2}">,
// CHECK-SAME: ["CHIP", 2]
// CHECK-SAME: fopt.chip = 2
// CHECK-NOT: {{(hbm|hbmv|dm|tu)<[a-z0-9]+, <"1">}}
// CHECK: return

#id = affine_map<(d0, d1) -> (d0, d1)>
module attributes {furiosa.chips = 2 : i64} {
  func.func @accumulate(%x: tensor<64x64xf32>, %out: tensor<64x64xf32>) -> tensor<64x64xf32>
      attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x, %out : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %sum : tensor<64x64xf32>
  }
}
