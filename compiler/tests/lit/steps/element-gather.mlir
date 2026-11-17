// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// tl.gather by data reads x[i, idx[i, j] mod 64]: one 4-byte element per address, which no DMA moves (8-byte units).
// The address a gathers the 8-byte row a >> 1 of x seen as [2048, 2]; copies take each lane and a select by the
// lane a & 1 picks it. The remainder by 64 expands to shifts and masks.

// CHECK-LABEL: func.func private @pick__s0
// CHECK-NOT: arith.remsi
// CHECK: arith.andi %{{.*}}, %c-64_i32
// CHECK: arith.shrsi %{{.*}}, %c1_i32
// CHECK-LABEL: func.func private @pick__s1
// CHECK: tensor.extract %{{.*}} : tensor<2048x2xi32>
// CHECK-LABEL: func.func private @pick__s2
// CHECK: tensor.extract_slice %{{.*}}[0, 0, 0] [64, 64, 1] [1, 1, 1]
// CHECK-LABEL: func.func private @pick__s3
// CHECK: tensor.extract_slice %{{.*}}[0, 0, 1] [64, 64, 1] [1, 1, 1]
// CHECK-LABEL: func.func private @pick__s4
// CHECK: %[[LANE:.*]] = arith.andi %{{.*}}, %c1_i32
// CHECK: %[[ODD:.*]] = arith.cmpi eq, %[[LANE]], %c1_i32
// CHECK: arith.select %[[ODD]], %in_1, %in_0
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @pick(%x: tensor<64x64xi32>, %idx: tensor<64x64xi32>, %o: tensor<64x64xi32>) -> tensor<64x64xi32> attributes {furiosa.written = [2]} {
    %c64 = arith.constant 64 : i32
    %e = tensor.empty() : tensor<64x64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%idx : tensor<64x64xi32>) outs(%e : tensor<64x64xi32>) {
    ^bb0(%k: i32, %out: i32):
      %i = linalg.index 0 : index
      %m = arith.remsi %k, %c64 : i32
      %j = arith.index_cast %m : i32 to index
      %v = tensor.extract %x[%i, %j] : tensor<64x64xi32>
      linalg.yield %v : i32
    } -> tensor<64x64xi32>
    return %r : tensor<64x64xi32>
  }
}
