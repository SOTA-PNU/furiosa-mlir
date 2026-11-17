// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=arith.select --implicit-check-not=arith.cmpi

// A generic that only chooses between two reads by an index comparison one loop decides splits at that loop's
// threshold into a copy per run, inserted into the result. tl.cat: n = 64 i + j < 2048 is i < 32, and over each
// run the read (64 i + j) mod 2048 is linear.

// CHECK-LABEL: func.func @cat
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<32x64xf32>)
// CHECK: %[[LO:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [32, 64] [1, 1]
// CHECK: linalg.generic {{.*}} outs(%{{.*}} : tensor<32x64xf32>)
// CHECK: tensor.insert_slice %{{.*}} into %[[LO]][32, 0] [32, 64] [1, 1]

// tl.join: the last loop (of 2) picks the side; each side, read along the first two loops, fills one column.

// CHECK-LABEL: func.func @join
// CHECK: %[[A:.*]] = tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0] [16, 64, 1] [1, 1, 1]
// CHECK: tensor.insert_slice %{{.*}} into %[[A]][0, 0, 1] [16, 64, 1] [1, 1, 1]
#id = affine_map<(d0, d1) -> (d0, d1)>
#half = affine_map<(d0, d1) -> ((d0 * 64 + d1) mod 2048)>
#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#side = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @cat(%a: tensor<2048xf32>, %b: tensor<2048xf32>, %o: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {furiosa.written = [2]} {
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %e = tensor.empty() : tensor<64x64xf32>
    %r = linalg.generic {indexing_maps = [#half, #half, #id], iterator_types = ["parallel", "parallel"]} ins(%a, %b : tensor<2048xf32>, tensor<2048xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %m = arith.muli %i, %c64 : index
      %n = arith.addi %m, %j : index
      %lt = arith.cmpi slt, %n, %c2048 : index
      %s = arith.select %lt, %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %r : tensor<64x64xf32>
  }
  func.func @join(%a: tensor<16x64xf32>, %b: tensor<16x64xf32>, %o: tensor<16x64x2xf32>) -> tensor<16x64x2xf32> attributes {furiosa.written = [2]} {
    %c0 = arith.constant 0 : index
    %e = tensor.empty() : tensor<16x64x2xf32>
    %r = linalg.generic {indexing_maps = [#side, #side, #id3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%a, %b : tensor<16x64xf32>, tensor<16x64xf32>) outs(%e : tensor<16x64x2xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %k = linalg.index 2 : index
      %first = arith.cmpi eq, %k, %c0 : index
      %s = arith.select %first, %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<16x64x2xf32>
    return %r : tensor<16x64x2xf32>
  }
}
