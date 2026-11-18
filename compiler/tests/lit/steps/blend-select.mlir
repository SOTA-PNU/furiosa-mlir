// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=arith.select

// A select keyed by the column alone (tl.sort's direction bit) between computed values, in a generic no class maps
// as it is (the row-wise class streams a select's key and computes nothing along the columns alone), becomes the
// bitwise blend b ^ ((a ^ b) & m) under the mask m = key ? ~0 : 0, a table along the columns once hoisted.

// CHECK-LABEL: func.func @by_column
// CHECK: %[[M:.*]] = arith.constant dense<[-1, 0, -1, 0
// CHECK: linalg.generic {{.*}} ins(%{{.*}}, %{{.*}}, %[[M]] :
// CHECK: %[[LO:.*]] = arith.minimumf
// CHECK: %[[HI:.*]] = arith.maximumf
// CHECK: %[[A:.*]] = arith.bitcast %[[HI]] : f32 to i32
// CHECK: %[[B:.*]] = arith.bitcast %[[LO]] : f32 to i32
// CHECK: %[[D:.*]] = arith.xori %[[A]], %[[B]] : i32
// CHECK: %[[AND:.*]] = arith.andi %[[D]], %{{.*}} : i32
// CHECK: %[[R:.*]] = arith.xori %[[B]], %[[AND]] : i32
// CHECK: arith.bitcast %[[R]] : i32 to f32
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @by_column(%x: tensor<64x64xf32>, %y: tensor<64xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %e = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#id, #row, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<64x64xf32>, tensor<64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %j = linalg.index 1 : index
      %bit = arith.andi %j, %c1 : index
      %even = arith.cmpi eq, %bit, %c0 : index
      %lo = arith.minimumf %a, %b : f32
      %hi = arith.maximumf %a, %b : f32
      %s = arith.select %even, %hi, %lo : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
}
