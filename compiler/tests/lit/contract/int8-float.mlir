// RUN: furiosa-mlir-opt --furiosa-normalize-contractions %s | FileCheck %s

// An int8 contraction (i32 sums of int8 products) over K = 64 <= 1024 runs on the f32 path, exactly: the operands
// convert to f32 and the sums, rounded, back to i32, in elementwise generics of their own. One over K = 2048 would
// not be exact in f32 and stays as it is.

// CHECK-LABEL: func.func @short
// CHECK: arith.sitofp %{{.*}} : i8 to f32
// CHECK: arith.sitofp %{{.*}} : i8 to f32
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: %[[R:.*]] = math.round %{{.*}} : f32
// CHECK: arith.fptosi %[[R]] : f32 to i32
// CHECK-LABEL: func.func @long
// CHECK-NOT: arith.sitofp
// CHECK: arith.muli
#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#y = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @short(%a: tensor<16x64xi8>, %b: tensor<32x64xi8>, %o: tensor<16x32xi32>) -> tensor<16x32xi32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16x32xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<16x32xi32>) -> tensor<16x32xi32>
    %r = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<16x64xi8>, tensor<32x64xi8>) outs(%f : tensor<16x32xi32>) {
    ^bb0(%p: i8, %q: i8, %acc: i32):
      %pe = arith.extsi %p : i8 to i32
      %qe = arith.extsi %q : i8 to i32
      %m = arith.muli %pe, %qe : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<16x32xi32>
    return %r : tensor<16x32xi32>
  }
  func.func @long(%a: tensor<16x2048xi8>, %b: tensor<32x2048xi8>, %o: tensor<16x32xi32>) -> tensor<16x32xi32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<16x32xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<16x32xi32>) -> tensor<16x32xi32>
    %r = linalg.generic {indexing_maps = [#x, #w, #y], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<16x2048xi8>, tensor<32x2048xi8>) outs(%f : tensor<16x32xi32>) {
    ^bb0(%p: i8, %q: i8, %acc: i32):
      %pe = arith.extsi %p : i8 to i32
      %qe = arith.extsi %q : i8 to i32
      %m = arith.muli %pe, %qe : i32
      %s = arith.addi %acc, %m : i32
      linalg.yield %s : i32
    } -> tensor<16x32xi32>
    return %r : tensor<16x32xi32>
  }
}
