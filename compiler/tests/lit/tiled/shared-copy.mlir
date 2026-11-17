// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s --check-prefix=STAGE
// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt --furiosa-dma %s | FileCheck %s

// y[p, c] = max(x[p - 1, c], x[p + 1, c]) over a -inf padding: two reads of one buffer that differ only in
// their offsets, so two stages. furiosa-dma stages them from one padded copy holding both windows (rows -1 to
// 64): one allocation of the padded extent, filled once, the buffer copied in once.

// STAGE-COUNT-2: fopt.stage %arg0
// CHECK-LABEL: func.func @two_taps
// CHECK-COUNT-1: fopt.alloc_hbm : !fopt.hbm<f32, <"1">, <"S{{[0-9]+}}_P0{66}, S{{[0-9]+}}_P1{64}">>
// CHECK-NOT: fopt.alloc_hbm : !fopt.hbm<f32, <"1">, <"S{{[0-9]+}}_P0

#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @two_taps(%x: tensor<64x64xf32>, %y: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [1]} {
    %ninf = arith.constant 0xFF800000 : f32
    %p = tensor.pad %x low[1, 0] high[1, 0] {
    ^bb0(%a: index, %b: index):
      tensor.yield %ninf : f32
    } : tensor<64x64xf32> to tensor<66x64xf32>
    %lo = tensor.extract_slice %p[0, 0] [64, 64] [1, 1] : tensor<66x64xf32> to tensor<64x64xf32>
    %hi = tensor.extract_slice %p[2, 0] [64, 64] [1, 1] : tensor<66x64xf32> to tensor<64x64xf32>
    %e = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%lo, %hi : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %o: f32):
      %m = arith.maxnumf %a, %b : f32
      linalg.yield %m : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
}
