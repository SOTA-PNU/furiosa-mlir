// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=arith.select

// A reduction whose steps a program-dependent trip count guards (program p runs steps j < 4 - 2p of a loop raised
// over 4 steps for both) splits by program: program 0 reduces over all 4 steps, program 1 over its first 2 (its
// input sliced to them), each from its slice of the initial value, with no guard and no discarded step.

// CHECK-LABEL: func.func @trips
// CHECK: %[[I0:.*]] = tensor.extract_slice %arg1[0, 0] [1, 8] [1, 1]
// CHECK: %[[X0:.*]] = tensor.extract_slice %arg0[0, 0, 0] [1, 8, 4] [1, 1, 1]
// CHECK: linalg.generic {{.*}} ins(%[[X0]] : tensor<1x8x4xf32>) outs(%[[I0]] : tensor<1x8xf32>)
// CHECK: %[[R0:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [1, 8] [1, 1]
// CHECK: %[[I1:.*]] = tensor.extract_slice %arg1[1, 0] [1, 8] [1, 1]
// CHECK: %[[X1:.*]] = tensor.extract_slice %arg0[1, 0, 0] [1, 8, 2] [1, 1, 1]
// CHECK: linalg.generic {{.*}} ins(%[[X1]] : tensor<1x8x2xf32>) outs(%[[I1]] : tensor<1x8xf32>)
// CHECK: tensor.insert_slice %{{.*}} into %[[R0]][1, 0] [1, 8] [1, 1]
#x = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#y = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @trips(%x: tensor<2x8x4xf32>, %acc: tensor<2x8xf32>, %o: tensor<2x8xf32>) -> tensor<2x8xf32> attributes {furiosa.written = [2]} {
    %c4 = arith.constant 4 : index
    %cm2 = arith.constant -2 : index
    %r = linalg.generic {indexing_maps = [#x, #y], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<2x8x4xf32>) outs(%acc : tensor<2x8xf32>) {
    ^bb0(%v: f32, %out: f32):
      %p = linalg.index 0 : index
      %j = linalg.index 2 : index
      %m = arith.muli %p, %cm2 : index
      %t = arith.addi %m, %c4 : index
      %on = arith.cmpi slt, %j, %t : index
      %s = arith.addf %out, %v : f32
      %g = arith.select %on, %s, %out : f32
      linalg.yield %g : f32
    } -> tensor<2x8xf32>
    return %r : tensor<2x8xf32>
  }
}
