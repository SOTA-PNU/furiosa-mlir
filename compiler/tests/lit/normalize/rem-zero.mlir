// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=arith.remsi

// A loop's step test on index values, (u - lb) rem 8 == 0, has no Vector Engine op: a remainder by a power of two
// m compared with 0 is its low bits, (u - lb) & 7 == 0, exact in any width and for negative values too.

// CHECK-LABEL: func.func @step
// CHECK: %[[L:.*]] = arith.andi %{{.*}}, %c7 : index
// CHECK: arith.cmpi eq, %[[L]], %c0 : index
#x = affine_map<(d0, d1) -> (d0, d1)>
#s = affine_map<(d0, d1) -> (d0)>
module {
  func.func @step(%x: tensor<4x64xf32>, %lb: tensor<4xi32>, %o: tensor<4xf32>) -> tensor<4xf32> attributes {furiosa.written = [2]} {
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %z = arith.constant -0.000000e+00 : f32
    %r = linalg.generic {indexing_maps = [#x, #s, #s], iterator_types = ["parallel", "reduction"]} ins(%x, %lb : tensor<4x64xf32>, tensor<4xi32>) outs(%o : tensor<4xf32>) {
    ^bb0(%v: f32, %l: i32, %acc: f32):
      %u = linalg.index 1 : index
      %li = arith.index_cast %l : i32 to index
      %off = arith.subi %u, %li : index
      %rem = arith.remsi %off, %c8 : index
      %step = arith.cmpi eq, %rem, %c0 : index
      %t = arith.select %step, %v, %z : f32
      %s = arith.addf %acc, %t : f32
      linalg.yield %s : f32
    } -> tensor<4xf32>
    return %r : tensor<4xf32>
  }
}
