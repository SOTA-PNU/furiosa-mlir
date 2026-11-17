// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s

// tl.argmax as the raise seeds it (no identity): the accumulators from element 0, then a reduction over elements
// 1.. whose combiner selects both by value order with an index tie-break. It becomes the greatest value m over
// all the elements, then the least index among the elements equal to m, both from their identities; the index
// term moves with the elements (index - 1 + 1 over the whole row).

// CHECK-LABEL: func.func @argmax
// CHECK: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[F:.*]] = linalg.fill ins(%[[NINF]] : f32)
// CHECK: %[[M:.*]] = linalg.generic {{.*}} iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<64x64xf32>) outs(%[[F]] : tensor<64xf32>)
// CHECK: arith.maximumf
// CHECK: %[[BIG:.*]] = arith.constant 2147483647 : i32
// CHECK: %[[G:.*]] = linalg.fill ins(%[[BIG]] : i32)
// CHECK: linalg.generic {{.*}} ins(%arg0, %[[M]] : tensor<64x64xf32>, tensor<64xf32>) outs(%[[G]] : tensor<64xi32>)
// CHECK: linalg.index 1
// CHECK: arith.cmpf oeq
// CHECK: arith.select
// CHECK: arith.minsi
#seed = affine_map<(d0) -> (d0, 0)>
#row = affine_map<(d0) -> (d0)>
#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d0)>
module {
  func.func @argmax(%x: tensor<64x64xf32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [1]} {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %ev = tensor.empty() : tensor<64xf32>
    %ei = tensor.empty() : tensor<64xi32>
    %first = tensor.extract_slice %x[0, 0] [64, 1] [1, 1] : tensor<64x64xf32> to tensor<64x1xf32>
    %seed:2 = linalg.generic {indexing_maps = [#seed, #row, #row], iterator_types = ["parallel"]} ins(%first : tensor<64x1xf32>) outs(%ev, %ei : tensor<64xf32>, tensor<64xi32>) {
    ^bb0(%in: f32, %out: f32, %out_1: i32):
      linalg.yield %in, %c0_i32 : f32, i32
    } -> (tensor<64xf32>, tensor<64xi32>)
    %rest = tensor.extract_slice %x[0, 1] [64, 63] [1, 1] : tensor<64x64xf32> to tensor<64x63xf32>
    %r:2 = linalg.generic {indexing_maps = [#in, #out, #out], iterator_types = ["parallel", "reduction"]} ins(%rest : tensor<64x63xf32>) outs(%seed#0, %seed#1 : tensor<64xf32>, tensor<64xi32>) {
    ^bb0(%in: f32, %out: f32, %out_1: i32):
      %j = linalg.index 1 : index
      %j1 = arith.addi %j, %c1 : index
      %ji = arith.index_cast %j1 : index to i32
      %eq = arith.cmpf oeq, %out, %in : f32
      %lt = arith.cmpi slt, %out_1, %ji : i32
      %tie = arith.andi %eq, %lt : i1
      %gt = arith.cmpf ogt, %out, %in : f32
      %keep = arith.ori %gt, %tie : i1
      %v = arith.select %keep, %out, %in : f32
      %i = arith.select %keep, %out_1, %ji : i32
      linalg.yield %v, %i : f32, i32
    } -> (tensor<64xf32>, tensor<64xi32>)
    return %r#1 : tensor<64xi32>
  }
}
