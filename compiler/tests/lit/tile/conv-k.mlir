// RUN: furiosa-mlir-opt --furiosa-tile %s | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s --check-prefix=FOPT

// furiosa-tile turns the plan's K tiles into a K loop: a 3x3 convolution with 128 input channels (VGG conv3)
// contracts K = 9 x 128 in three passes of 3 taps (384 fit a TRF lane with the column groups the plan
// chose), so the loop tiles the kernel row (dim 3) one at a time, accumulating in its iteration argument.
// Mapping folds the loop back into the plan's K tiles (KF / 384).

// CHECK-LABEL: func.func @conv_k
// CHECK: scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<112x112x128xf32>)
// CHECK: tensor.extract_slice %{{.*}}[0, %[[K]], 0, 0] [1, 112, 114, 128]
// CHECK: tensor.extract_slice %{{.*}}[0, 0, %[[K]], 0] [128, 128, 1, 3]
// CHECK: } {furiosa.tile_dim = 3 : i64}

// FOPT-COUNT-3: index <"KF{1152} / 384"> len 1 start

#x = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 + d3, d1 + d4, d5)>
#w = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d5, d3, d4)>
#y = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#r3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#c = affine_map<(d0, d1, d2, d3) -> (d3)>
#o = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @conv_k(%arg0: tensor<1x112x112x128xf32>, %arg1: tensor<128x128x3x3xf32>, %arg2: tensor<128xf32>, %arg3: tensor<1x112x112x128xf32>) -> tensor<1x112x112x128xf32> attributes {furiosa.written = [3]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x112x112x128xf32>
    %1 = tensor.empty() : tensor<112x112x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<112x112x128xf32>) -> tensor<112x112x128xf32>
    %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%a: index, %b: index, %c: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x112x112x128xf32> to tensor<1x114x114x128xf32>
    %3 = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<1x114x114x128xf32>, tensor<128x128x3x3xf32>) outs(%2 : tensor<112x112x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<112x112x128xf32>
    %4 = linalg.generic {indexing_maps = [#r3, #c, #o], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg2 : tensor<112x112x128xf32>, tensor<128xf32>) outs(%0 : tensor<1x112x112x128xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.maxnumf %5, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<1x112x112x128xf32>
    return %4 : tensor<1x112x112x128xf32>
  }
}
