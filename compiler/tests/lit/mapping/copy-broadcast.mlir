// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// The copy class maps a generic that only broadcasts its input (tl.broadcast_to of a row) as one HBM-to-HBM DMA:
// the input's unit dims (read at index 0) are left out of its layout, and the output axis the input lacks repeats it.

// CHECK-LABEL: func.func @rows
// CHECK: %[[SRC:.*]] = fopt.view %arg0 : !fopt.hbm<f32, <"1">, <"D2{64}">> -> !fopt.hbmv<f32, <"1">, <"D2{64}">, false>
// CHECK: %[[DST:.*]] = fopt.view %arg1 : !fopt.hbm<f32, <"1">, <"B1{32}, D2{64}">> -> !fopt.hbmv<f32, <"1">, <"B1{32}, D2{64}">, true>
// CHECK: fopt.hbm_to_hbm_view %[[SRC]], %[[DST]]
#in = affine_map<(d0, d1, d2) -> (0, 0, d2)>
#id = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  func.func @rows(%x: tensor<1x1x64xf32>, %o: tensor<1x32x64xf32>) -> tensor<1x32x64xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<1x32x64xf32>
    %r = linalg.generic {indexing_maps = [#in, #id], iterator_types = ["parallel", "parallel", "parallel"]} ins(%x : tensor<1x1x64xf32>) outs(%e : tensor<1x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<1x32x64xf32>
    return %r : tensor<1x32x64xf32>
  }
}
