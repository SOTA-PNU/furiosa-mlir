// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not="tensor.extract "

// A loop the raise kept at the top level (here a running sum over two tiles of x, as in an argmax's running
// maximum) stays a loop the host runs: its body generic is lifted over the iterations so furiosa-structure
// promotes the read the induction variable addressed (no gather-form extract is left), then tiled back to one
// iteration, which reads x's row at the iteration and is split into a step called once per iteration.

// CHECK-LABEL: func.func @k
// CHECK: %[[X:.*]] = tensor.expand_shape %arg0 {{.*}} into tensor<2x4xf32>
// CHECK: scf.for %[[I:.*]] = %{{.*}} iter_args
// CHECK: tensor.extract_slice %[[X]][%[[I]], 0] [1, 4]
// CHECK: func.call @k__s0
// CHECK: } {furiosa.steps}
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
