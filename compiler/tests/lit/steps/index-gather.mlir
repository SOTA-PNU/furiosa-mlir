// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s
// RUN: furiosa-mlir-opt --furiosa-split-steps %s | furiosa-mlir-opt --furiosa-steps-plan | FileCheck %s --check-prefix=PLAN

// Nearest upsampling reads x[i, floor(j * 0.5)]: an address of the loop indices alone that no indexing map takes
// (a float floor). It gathers like a data-dependent one, from the table of its addresses (0, 0, 1, 1, 2, ...): the
// 8-byte row a >> 1 of x seen as [75, 2], a copy per lane, a select by the lane a & 1. The gather DMA takes whole
// 256s of rows past 256: the 300 addresses are inserted into 512 zeros, a copy step (the row-wise class pads no
// row tiles).

// The plan's step modules take the tables as trailing arguments marked furiosa.table.
// PLAN: func.func @nearest__s0(%arg0: tensor<6x50xi32> {{.*}}, %arg1: tensor<6x50xi32> {furiosa.table}

// CHECK-LABEL: func.func private @nearest__s0
// CHECK: arith.constant dense<"0x0000000000000000010000000100000002000000
// CHECK-LABEL: func.func private @nearest__s1
// CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0] [300] [1] : tensor<300xi32> into tensor<512xi32>
// CHECK-LABEL: func.func private @nearest__s2
// CHECK: arith.shrsi %{{.*}}, %c1_i32
// CHECK-LABEL: func.func private @nearest__s3
// CHECK: tensor.extract %{{.*}} : tensor<75x2xf32>
// CHECK-LABEL: func.func private @nearest__s6
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %c1_i32
// CHECK: %[[ODD:.*]] = arith.cmpi eq, %[[LANE]], %c1_i32
// CHECK: arith.select %[[ODD]], %in_1, %in_0
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @nearest(%x: tensor<6x25xf32>, %y: tensor<6x50xf32>) -> tensor<6x50xf32> attributes {furiosa.written = [1]} {
    %half = arith.constant 5.000000e-01 : f32
    %c25 = arith.constant 25 : index
    %flat = tensor.collapse_shape %x [[0, 1]] : tensor<6x25xf32> into tensor<150xf32>
    %e = tensor.empty() : tensor<6x50xf32>
    %r = linalg.generic {indexing_maps = [#id], iterator_types = ["parallel", "parallel"]} outs(%e : tensor<6x50xf32>) {
    ^bb0(%out: f32):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %ji = arith.index_cast %j : index to i32
      %jf = arith.sitofp %ji : i32 to f32
      %s = arith.mulf %jf, %half : f32
      %w = arith.fptosi %s : f32 to i32
      %wi = arith.index_cast %w : i32 to index
      %row = arith.muli %i, %c25 : index
      %a = arith.addi %row, %wi : index
      %v = tensor.extract %flat[%a] : tensor<150xf32>
      linalg.yield %v : f32
    } -> tensor<6x50xf32>
    return %r : tensor<6x50xf32>
  }
}
