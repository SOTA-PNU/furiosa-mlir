// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// tl.sort's compare-and-swap, fused with the xor reduction that finds the next partner: the term's select keyed by
// (x > y) != flip nests into selects keyed by one condition each, though its generic reduces; the combiner (the xor
// with the accumulator) is not a select and stays.

// CHECK-LABEL: func.func @swap_then_xor
// CHECK-NOT: arith.cmpi ne, %{{.*}} : i1
// CHECK: %[[GT:.*]] = arith.cmpf ogt
// CHECK-DAG: %[[LO:.*]] = arith.select %[[GT]], %{{.*}}, %{{.*}} : f32
// CHECK-DAG: %[[HI:.*]] = arith.select %[[GT]], %{{.*}}, %{{.*}} : f32
// CHECK: arith.select %{{.*}}, %[[HI]], %[[LO]] : f32
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @swap_then_xor(%x: tensor<64x2xf32>, %y: tensor<64x2xf32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [2]} {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<64xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<64xi32>) -> tensor<64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #row], iterator_types = ["parallel", "reduction"]} ins(%x, %y : tensor<64x2xf32>, tensor<64x2xf32>) outs(%z : tensor<64xi32>) {
    ^bb0(%a: f32, %b: f32, %acc: i32):
      %j = linalg.index 1 : index
      %ji = arith.index_cast %j : index to i32
      %flip = arith.cmpi ne, %ji, %c0 : i32
      %gt = arith.cmpf ogt, %a, %b : f32
      %swap = arith.cmpi ne, %gt, %flip : i1
      %v = arith.select %swap, %b, %a : f32
      %bits = arith.bitcast %v : f32 to i32
      %n = arith.xori %bits, %acc : i32
      linalg.yield %n : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}
