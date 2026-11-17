// RUN: furiosa-mlir-opt --furiosa-tile %s | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma %s | FileCheck %s --check-prefix=FOPT

// furiosa-tile tiles a kernel by the tiled lowering's plan: a 3x3 convolution over 224x224x64 (VGG conv1_2)
// splits its output rows into 4 tiles of 56. The loop computes each tile's contraction from a 58-row window of
// the padded input (tile and fuse) and its epilogue, and inserts it into the result; the pad stays whole.
// Mapping the tiled kernel folds the loop back into the plan's tile digit (A0 / 56).

// CHECK-LABEL: func.func @conv
// CHECK: tensor.pad
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<1x224x224x64xf32>)
// CHECK: tensor.extract_slice %{{.*}}[0, %[[I]], 0, 0] [1, 58, 226, 64]
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<56x224x64xf32>)
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<1x56x224x64xf32>)
// CHECK: tensor.insert_slice %{{.*}} into %[[ACC]][0, %[[I]], 0, 0] [1, 56, 224, 64]

// FOPT: fopt.tile %{{.*}} index <"A0{224} / 56"> len 1 start 3

#x = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, d0 + d3, d1 + d4, d5)>
#w = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d5, d3, d4)>
#y = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>
#r3 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#c = affine_map<(d0, d1, d2, d3) -> (d3)>
#o = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @conv(%arg0: tensor<1x224x224x64xf32>, %arg1: tensor<64x64x3x3xf32>, %arg2: tensor<64xf32>, %arg3: tensor<1x224x224x64xf32>) -> tensor<1x224x224x64xf32> attributes {furiosa.written = [3]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x224x224x64xf32>
    %1 = tensor.empty() : tensor<224x224x64xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<224x224x64xf32>) -> tensor<224x224x64xf32>
    %padded = tensor.pad %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0] {
    ^bb0(%a: index, %b: index, %c: index, %d: index):
      tensor.yield %cst : f32
    } : tensor<1x224x224x64xf32> to tensor<1x226x226x64xf32>
    %3 = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%padded, %arg1 : tensor<1x226x226x64xf32>, tensor<64x64x3x3xf32>) outs(%2 : tensor<224x224x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %out, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<224x224x64xf32>
    %4 = linalg.generic {indexing_maps = [#r3, #c, #o], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3, %arg2 : tensor<224x224x64xf32>, tensor<64xf32>) outs(%0 : tensor<1x224x224x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      %6 = arith.maxnumf %5, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<1x224x224x64xf32>
    return %4 : tensor<1x224x224x64xf32>
  }
}
