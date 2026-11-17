// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// Rows whose length is not a multiple of 16 (14 rows of 33; 462 elements regroup into no rows of 16-multiples)
// pad to whole packets: the column axis is H{48} in DM and in the streams, and HBM holds its first 33 (H = 33).
// The DMAs move rows only in whole 8-byte units (33 i32 are 132 bytes), so each input is copied (byte-granular,
// HBM to HBM) into the columns of a padded scratch and moves to DM from there, and the result goes back the same
// way. Elementwise ops never read the padding; a reduction reads it, so a float input it reduces pads with its
// identity (other reductions over padded rows are refused).

// CHECK-LABEL: func.func @add
// CHECK-SAME: !fopt.hbm<i32, <"1">, <"R{14}, H{48} = 33">>
// CHECK: %[[S:.*]] = fopt.alloc_hbm : !fopt.hbm<i32, <"1">, <"R{14}, H{48}">>
// CHECK: %[[C:.*]] = fopt.tile {{.*}} index <"H{48}"> len 33 start 0 : {{.*}} -> !fopt.hbmv<i32, <"1">, <"R{14}, H{48} = 33 #{!} 48">, true>
// CHECK: fopt.hbm_to_hbm_view %{{.*}}, %[[C]]
// CHECK: fopt.to_dm %[[S]] : {{.*}} -> !fopt.dm<i32, <"1">, <"1 # 2">, <"R{14} # 256">, <"H{48}">>
// CHECK: fopt.fetch {{.*}} -> !fopt.tu<i32, <"1">, <"1 # 2">, <"R{14} # 256">, <"H{48} / 8, I{2}">, <"H{48} % 8">>
// CHECK: %[[O:.*]] = fopt.alloc_hbm : !fopt.hbm<i32, <"1">, <"R{14}, H{48} = 33">>
// CHECK: fopt.to_hbm {{.*}} -> !fopt.hbm<i32, <"1">, <"R{14}, H{48}">>
// CHECK: %[[I:.*]] = fopt.tile {{.*}} index <"H{48}"> len 33 start 0 : {{.*}} -> !fopt.hbmv<i32, <"1">, <"R{14}, H{48} = 33 # 48">, false>
// CHECK: fopt.hbm_to_hbm_view %[[I]], %{{.*}} : {{.*}}, !fopt.hbmv<i32, <"1">, <"R{14}, H{48} = 33">, true>
// CHECK: return %[[O]]

// CHECK-LABEL: func.func @row_max
// CHECK: fopt.memset %{{.*}}, 0xFF800000
// CHECK: fopt.vector_intra_slice_reduce "Max"
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
module {
  func.func @add(%x: tensor<14x33xi32>, %y: tensor<14x33xi32>, %o: tensor<14x33xi32>) -> tensor<14x33xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<14x33xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<14x33xi32>, tensor<14x33xi32>) outs(%e : tensor<14x33xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<14x33xi32>
    return %r : tensor<14x33xi32>
  }
  func.func @row_max(%x: tensor<14x33xbf16>, %o: tensor<14xf32>) -> tensor<14xf32> attributes {furiosa.written = [1]} {
    %ninf = arith.constant 0xFF800000 : f32
    %e = tensor.empty() : tensor<14xf32>
    %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<14xf32>) -> tensor<14xf32>
    %r = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<14x33xbf16>) outs(%f : tensor<14xf32>) {
    ^bb0(%a: bf16, %acc: f32):
      %c = arith.extf %a : bf16 to f32
      %m = arith.maximumf %acc, %c : f32
      linalg.yield %m : f32
    } -> tensor<14xf32>
    return %r : tensor<14xf32>
  }
}
