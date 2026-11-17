// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=scf.for --implicit-check-not="tensor.extract "

// A loop the raise kept at the top level (here a running sum over two tiles of x, as in an argmax's running
// maximum) unrolls into host-launched iterations; furiosa-structure then promotes the reads the induction
// variable addressed, so no gather-form extract is left.

// CHECK-LABEL: func.func @k
// CHECK: tensor.extract_slice %arg0[0] [4] [1]
// CHECK: tensor.extract_slice %arg0[4] [4] [1]
#id = affine_map<(d0) -> (d0)>
module {
  func.func @k(%x: tensor<8xf32>, %o: tensor<4xf32>) -> tensor<4xf32> attributes {furiosa.written = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<4xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<4xf32>) -> tensor<4xf32>
    %r = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %f) -> (tensor<4xf32>) {
      %g = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%acc : tensor<4xf32>) outs(%e : tensor<4xf32>) {
      ^bb0(%a: f32, %out: f32):
        %j = linalg.index 0 : index
        %b = arith.muli %i, %c4 : index
        %k = arith.addi %b, %j : index
        %v = tensor.extract %x[%k] : tensor<8xf32>
        %s = arith.addf %a, %v : f32
        linalg.yield %s : f32
      } -> tensor<4xf32>
      scf.yield %g : tensor<4xf32>
    }
    return %r : tensor<4xf32>
  }
}
