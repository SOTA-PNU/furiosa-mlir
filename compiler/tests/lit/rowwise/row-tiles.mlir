// RUN: furiosa-mlir-opt --furiosa-tile %s | FileCheck %s --check-prefix=LOOP
// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s --check-prefix=FOPT
// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s --check-prefix=FOPT

// Rows beyond the 256 slices (here 32 x 16 rows of a row max) take windows of the outermost row dim: 16 of its
// rows, which with the 16 rows inside fill the slices. One decision (Rowwise's rowTile) serves both paths:
// furiosa-tile makes it a loop over the root generic, and the row-wise class maps that tiled kernel to the same
// fopt as the untiled one, a pass sequence per window over tile views.

// LOOP-LABEL: func.func @rowmax
// LOOP: scf.for %{{.*}} = %c0 to %c32 step %c16 iter_args
// LOOP: linalg.generic {{.*}} ins(%{{.*}} : tensor<16x16x64xf32>)

// FOPT-LABEL: func.func @rowmax
// FOPT-SAME: !fopt.hbm<f32, <"1">, <"R{32}, R1{16}">> {fopt.arg = 1 : i64, fopt.inout
// FOPT: fopt.tile %{{.*}} index <"R{32}"> len 16 start 0 : {{.*}} false>
// FOPT: fopt.vector_intra_slice_reduce "Max"
// FOPT: fopt.tile %{{.*}} index <"R{32}"> len 16 start 0 : {{.*}} true>
// FOPT: fopt.tile %{{.*}} index <"R{32}"> len 16 start 16 : {{.*}} false>
// FOPT: fopt.vector_intra_slice_reduce "Max"
// FOPT: fopt.tile %{{.*}} index <"R{32}"> len 16 start 16 : {{.*}} true>
#x = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#r = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @rowmax(%x: tensor<32x16x64xf32>, %o: tensor<32x16xf32>) -> tensor<32x16xf32> attributes {furiosa.written = [1]} {
    %ninf = arith.constant 0xFF800000 : f32
    %e = tensor.empty() : tensor<32x16xf32>
    %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<32x16xf32>) -> tensor<32x16xf32>
    %m = linalg.generic {indexing_maps = [#x, #r], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<32x16x64xf32>) outs(%f : tensor<32x16xf32>) {
    ^bb0(%a: f32, %acc: f32):
      %y = arith.maximumf %acc, %a : f32
      linalg.yield %y : f32
    } -> tensor<32x16xf32>
    return %m : tensor<32x16xf32>
  }
}
