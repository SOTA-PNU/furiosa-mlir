// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=scf.for

// A kept loop whose iterations read a box the plan cannot view as one byte range (here a block of columns from
// each of two rows, as a row-wise logsumexp over column blocks does) unrolls instead of rolling: its iterations'
// boxes, at static offsets once the index arithmetic folds, are copy steps.

// CHECK-LABEL: func.func @k
// CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [2, 1, 4]
// CHECK: tensor.extract_slice %{{.*}}[0, 1, 0] [2, 1, 4]
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @k(%x: tensor<2x8xf32>, %o: tensor<2x4xf32>) -> tensor<2x4xf32> attributes {furiosa.written = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<2x4xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<2x4xf32>) -> tensor<2x4xf32>
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %f) -> (tensor<2x4xf32>) {
      %g = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%acc : tensor<2x4xf32>) outs(%e : tensor<2x4xf32>) {
      ^bb0(%a: f32, %out: f32):
        %row = linalg.index 0 : index
        %j = linalg.index 1 : index
        %b = arith.muli %i, %c4 : index
        %k = arith.addi %b, %j : index
        %v = tensor.extract %x[%row, %k] : tensor<2x8xf32>
        %s = arith.addf %a, %v : f32
        linalg.yield %s : f32
      } -> tensor<2x4xf32>
      scf.yield %g : tensor<2x4xf32>
    }
    return %r : tensor<2x4xf32>
  }
}
