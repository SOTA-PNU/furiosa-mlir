// RUN: furiosa-mlir-opt --furiosa-tile --split-input-file %s | FileCheck %s --check-prefix=LOOP
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt --split-input-file %s | FileCheck %s

// The slices hold windows of one row dim by the other row dims, each padded to a power of two (the NPU misplaces a
// slice digit inside the outermost otherwise).

// @padded_inner (tl.cumsum's form: a row's x masked by a table over [3 rows, 20 prefixes, 16 columns]): the 20
// rows inside the outermost pad to 32, the 3 outer ones fill the rest of the 256 slices.

// CHECK-LABEL: func.func @padded_inner
// CHECK: fopt.view_to_dm %{{.*}} -> !fopt.dm<i32, <"1">, <"1 # 2">, <"R{3} # 8, R1{20} # 32">, <"H{16}">>

// @windowed_inner (the same over [5, 300]): 300 rows padded (512) exceed the slices beside the 5, so the slices take
// windows of the 300 by the 5 padded to 8: 32 of them fill the slices, 30 divide the 300 (one device loop of 10, as
// furiosa-tile's loop), and x, which does not vary along the windowed dim, is read whole in each window.

// LOOP-LABEL: func.func @windowed_inner
// LOOP: scf.for %{{.*}} = %c0 to %c300 step %c30 iter_args

// CHECK-LABEL: func.func @windowed_inner
// CHECK: "fopt.for"() <{count = 10 : i64}>
// CHECK: fopt.view_to_dm %{{.*}} : !fopt.hbmv<i32, <"1">, <"R{5}, H{16}">, false> -> !fopt.dm<i32, <"1">, <"1 # 2">, <"R1{300} = 30 # 32, R{5} # 8">, <"H{16}">>
// CHECK: "fopt.tile_at"({{.*}}) <{coeffs = array<i64: 30>, index = #fopt.m<"R1{300}">, len = 30 : i64, start = 0 : i64}>
// CHECK: fopt.vector_intra_slice_reduce "AddSat"
// CHECK: "fopt.tile_at"({{.*}}) <{coeffs = array<i64: 30>, index = #fopt.m<"R1{300}">, len = 30 : i64, start = 0 : i64}> {{.*}} -> !fopt.hbmv<i32, <"1">, <"R{5}, R1{300} = 30 #{!} 300">, true>

#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#t = affine_map<(d0, d1, d2) -> (d1, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @padded_inner(%x: tensor<3x16xi32>, %tab: tensor<20x16xi32>, %o: tensor<3x20xi32>) -> tensor<3x20xi32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<3x20xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<3x20xi32>) -> tensor<3x20xi32>
    %r = linalg.generic {indexing_maps = [#x, #t, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %tab : tensor<3x16xi32>, tensor<20x16xi32>) outs(%f : tensor<3x20xi32>) {
    ^bb0(%a: i32, %t: i32, %acc: i32):
      %on = arith.cmpi ne, %t, %zero : i32
      %v = arith.select %on, %a, %zero : i32
      %s = arith.addi %acc, %v : i32
      linalg.yield %s : i32
    } -> tensor<3x20xi32>
    return %r : tensor<3x20xi32>
  }
}

// -----

#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#t = affine_map<(d0, d1, d2) -> (d1, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @windowed_inner(%x: tensor<5x16xi32>, %tab: tensor<300x16xi32>, %o: tensor<5x300xi32>) -> tensor<5x300xi32> attributes {furiosa.written = [2]} {
    %zero = arith.constant 0 : i32
    %e = tensor.empty() : tensor<5x300xi32>
    %f = linalg.fill ins(%zero : i32) outs(%e : tensor<5x300xi32>) -> tensor<5x300xi32>
    %r = linalg.generic {indexing_maps = [#x, #t, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %tab : tensor<5x16xi32>, tensor<300x16xi32>) outs(%f : tensor<5x300xi32>) {
    ^bb0(%a: i32, %t: i32, %acc: i32):
      %on = arith.cmpi ne, %t, %zero : i32
      %v = arith.select %on, %a, %zero : i32
      %s = arith.addi %acc, %v : i32
      linalg.yield %s : i32
    } -> tensor<5x300xi32>
    return %r : tensor<5x300xi32>
  }
}
