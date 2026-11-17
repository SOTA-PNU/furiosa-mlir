// RUN: furiosa-mlir-opt --furiosa-split-steps --furiosa-steps-plan %s -o /dev/null | FileCheck %s

// The launch plan of a rolled kept loop (furiosa-steps-plan prints it): the loop runs `count` iterations of
// its body steps; a carried value starts in a table (the fill of its init) and takes each iteration's yield;
// a slice at the iteration is a view of the buffer, bytes [offset + i * iter_stride, + bytes); the kernel's
// result is the loop's final carried value, handed back as the written buffer.

// CHECK: "kernel":"k","returns":[{"buffer":1,"carried":0}]
// CHECK-SAME: "steps":[{"loop":{"carried":[{"const":0}],"count":2,"steps":[{"args":[{"carried":0},{"buffer":0,"bytes":16,"iter_stride":16,"offset":0},{"temp":0}],"func":"k__s0"
// CHECK-SAME: "yields":[{"temp":0}]
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
