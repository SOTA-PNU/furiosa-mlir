// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// Rows whose length is not a multiple of 16 (14 rows of 33; 462 elements regroup into no rows of 16-multiples)
// pad to whole packets: the column axis is H{48} in DM and in the streams, and HBM holds its first 33 (H = 33).
// The DMAs move rows only in whole 8-byte units (33 i32 are 132 bytes), so each input is copied (byte-granular,
// HBM to HBM) into the columns of a padded scratch and moves to DM from there, and the result goes back the same
// way. Elementwise ops never read the padding; a reduction reads it, so an input it reduces pads with its
// identity (an i32 one, its bits: INT_MIN for a max). A read padded past its last column (tensor.pad: a masked
// load's `other`) holds its own columns, the scratch filled with the pad first. One column (14 x 1) moves no whole
// packet: its 14 elements regroup into one row. A row value stored as bf16 crosses one 8-byte unit (4 bf16).

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
// CHECK-LABEL: func.func @pad_max
// CHECK-SAME: !fopt.hbm<i32, <"1">, <"R{14}, H{64} = 33">>
// CHECK: fopt.memset %{{.*}}, -0.000000e+00 : !fopt.dm<i32
// CHECK: fopt.tile {{.*}} index <"H{64}"> len 33 start 0
// CHECK: fopt.vector_intra_slice_reduce "Max" {{.*}} !fopt.tu<i32
// CHECK-LABEL: func.func @i32_max
// CHECK: fopt.memset %{{.*}}, -0.000000e+00 : !fopt.dm<i32
// CHECK: fopt.tile {{.*}} index <"H{48}"> len 33 start 0
// CHECK-LABEL: func.func @one_column
// CHECK-SAME: !fopt.hbm<i32, <"1">, <"H{16} = 14">>
// CHECK: fopt.tile {{.*}} index <"H{16}"> len 14 start 0
// CHECK-LABEL: func.func @bf16_rows
// CHECK: fopt.to_hbm {{.*}} -> !fopt.hbm<bf16, <"1">, <"R{14}, 1 # 4">>
// CHECK-LABEL: func.func @scalar_cast
// CHECK: fopt.vector_intra_slice_reduce "Max"
// CHECK: fopt.to_hbm {{.*}} -> !fopt.hbm<bf16, <"1">, <"R{1}, 1 # 4">>
#id = affine_map<(d0, d1) -> (d0, d1)>
#row = affine_map<(d0, d1) -> (d0)>
#v = affine_map<(d0) -> (0, d0)>
#s = affine_map<(d0) -> ()>
#o = affine_map<(d0) -> (d0)>
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
  func.func @pad_max(%x: tensor<14x33xi32>, %o: tensor<14xi32>) -> tensor<14xi32> attributes {furiosa.written = [1]} {
    %lo = arith.constant -2147483648 : i32
    %e = tensor.empty() : tensor<14xi32>
    %f = linalg.fill ins(%lo : i32) outs(%e : tensor<14xi32>) -> tensor<14xi32>
    %p = tensor.pad %x low[0, 0] high[0, 31] {
    ^bb0(%i: index, %j: index):
      tensor.yield %lo : i32
    } : tensor<14x33xi32> to tensor<14x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%p : tensor<14x64xi32>) outs(%f : tensor<14xi32>) {
    ^bb0(%a: i32, %acc: i32):
      %m = arith.maxsi %acc, %a : i32
      linalg.yield %m : i32
    } -> tensor<14xi32>
    return %r : tensor<14xi32>
  }
  func.func @i32_max(%x: tensor<14x33xi32>, %o: tensor<14xi32>) -> tensor<14xi32> attributes {furiosa.written = [1]} {
    %lo = arith.constant -2147483648 : i32
    %e = tensor.empty() : tensor<14xi32>
    %f = linalg.fill ins(%lo : i32) outs(%e : tensor<14xi32>) -> tensor<14xi32>
    %r = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<14x33xi32>) outs(%f : tensor<14xi32>) {
    ^bb0(%a: i32, %acc: i32):
      %m = arith.maxsi %acc, %a : i32
      linalg.yield %m : i32
    } -> tensor<14xi32>
    return %r : tensor<14xi32>
  }
  func.func @one_column(%x: tensor<14x1xi32>, %y: tensor<14x1xi32>, %o: tensor<14x1xi32>) -> tensor<14x1xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<14x1xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]} ins(%x, %y : tensor<14x1xi32>, tensor<14x1xi32>) outs(%e : tensor<14x1xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %z = arith.xori %a, %b : i32
      linalg.yield %z : i32
    } -> tensor<14x1xi32>
    return %r : tensor<14x1xi32>
  }
  func.func @bf16_rows(%x: tensor<14x64xbf16>, %o: tensor<14xbf16>) -> tensor<14xbf16> attributes {furiosa.written = [1]} {
    %zero = arith.constant 0.0 : bf16
    %e = tensor.empty() : tensor<14xbf16>
    %f = linalg.fill ins(%zero : bf16) outs(%e : tensor<14xbf16>) -> tensor<14xbf16>
    %r = linalg.generic {indexing_maps = [#id, #row], iterator_types = ["parallel", "reduction"]} ins(%x : tensor<14x64xbf16>) outs(%f : tensor<14xbf16>) {
    ^bb0(%a: bf16, %acc: bf16):
      %s = arith.addf %acc, %a : bf16
      linalg.yield %s : bf16
    } -> tensor<14xbf16>
    return %r : tensor<14xbf16>
  }
  func.func @scalar_cast(%x: tensor<1x1024xf32>, %o: tensor<1xbf16>) -> tensor<1xbf16> attributes {furiosa.written = [1]} {
    %ninf = arith.constant 0xFF800000 : f32
    %e = tensor.empty() : tensor<f32>
    %f = linalg.fill ins(%ninf : f32) outs(%e : tensor<f32>) -> tensor<f32>
    %r = linalg.generic {indexing_maps = [#v, #s], iterator_types = ["reduction"]} ins(%x : tensor<1x1024xf32>) outs(%f : tensor<f32>) {
    ^bb0(%a: f32, %acc: f32):
      %m = arith.maxnumf %acc, %a : f32
      linalg.yield %m : f32
    } -> tensor<f32>
    %e1 = tensor.empty() : tensor<1xbf16>
    %c = linalg.generic {indexing_maps = [#s, #o], iterator_types = ["parallel"]} ins(%r : tensor<f32>) outs(%e1 : tensor<1xbf16>) {
    ^bb0(%a: f32, %out: bf16):
      %t = arith.truncf %a : f32 to bf16
      linalg.yield %t : bf16
    } -> tensor<1xbf16>
    return %c : tensor<1xbf16>
  }
}
