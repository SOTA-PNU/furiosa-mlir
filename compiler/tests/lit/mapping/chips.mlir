// RUN: furiosa-mlir-opt --furiosa-shard %s | FileCheck %s --check-prefix=SHARD
// RUN: furiosa-mlir-opt --furiosa-shard --furiosa-tile --furiosa-linalg-to-fopt --furiosa-dma --furiosa-ve-fuse %s | FileCheck %s

// SPMD over two chips (furiosa.chips = 2): every kernel runs on both chips (#[device(chip = 2)]), each on its own
// copy of the buffers, with the CHIP axis as chip mapping. furiosa-shard splits the outermost parallel dim of a result
// whose blocks are boxes of one dim of every operand: the matmul's rows, so each chip reads its rows of x and all of
// w. Mapping gives the kernel the whole buffers, the same on every chip (chip mapping m![2]): it copies its block of x
// out of them, maps the block as on one chip, and gathers its block of the result into every chip's copy (the
// all-gather, a DMA across chips). A buffer the kernel reads and writes (in place) splits as its result does, each
// chip reading its own block (accumulate); read at another index (transposed), the kernel runs whole on every chip.
// A slice that keeps the split dim whole is a block of its source's block: a pool's window reads split by channel.

// SHARD-LABEL: func.func @mm
// SHARD-SAME: (%arg0: tensor<32x64xf32> {furiosa.shard = 0 : i64}, %arg1: tensor<64x64xf32>,
// SHARD-SAME: %arg2: tensor<32x64xf32> {furiosa.shard = 0 : i64})
// SHARD-SAME: -> (tensor<32x64xf32> {furiosa.shard = 0 : i64})
// SHARD: linalg.generic
// SHARD-SAME: ins({{.*}} : tensor<32x64xf32>, tensor<64x64xf32>) outs({{.*}} : tensor<32x64xf32>)
// SHARD-LABEL: func.func @accumulate
// SHARD-SAME: (%arg0: tensor<32x64xf32> {furiosa.shard = 0 : i64}, %arg1: tensor<32x64xf32> {furiosa.shard = 0 : i64})
// SHARD-SAME: -> (tensor<32x64xf32> {furiosa.shard = 0 : i64})
// SHARD-LABEL: func.func @transposed
// SHARD-SAME: (%arg0: tensor<64x64xf32>, %arg1: tensor<64x64xf32>) -> tensor<64x64xf32>
// SHARD-LABEL: func.func @pool
// SHARD-SAME: (%arg0: tensor<8x32xf32> {furiosa.shard = 1 : i64}, %arg1: tensor<7x32xf32> {furiosa.shard = 1 : i64})
// SHARD: tensor.extract_slice %arg0[0, 0] [7, 32] [1, 1] : tensor<8x32xf32> to tensor<7x32xf32>
// SHARD: tensor.extract_slice %arg0[1, 0] [7, 32] [1, 1] : tensor<8x32xf32> to tensor<7x32xf32>

// CHECK-LABEL: func.func @mm
// CHECK-SAME: (%arg0: !fopt.hbm<f32, <"2">, <"CHIP{2}, [[R:[A-Z0-9_]+]]{32}, [[C:[A-Z0-9_]+]]{64}">>
// CHECK-SAME: %arg1: !fopt.hbm<f32, <"CHIP{2}">,
// CHECK-SAME: -> (!fopt.hbm<f32, <"2">, <"CHIP{2}, [[OR:[A-Z0-9_]+]]{32}, [[OC:[A-Z0-9_]+]]{64}">>
// CHECK-SAME: fopt.chip = 2
// CHECK: %[[X:.*]] = fopt.alloc_hbm : !fopt.hbm<f32, <"CHIP{2}">, <"[[R]]{32}, [[C]]{64}">>
// CHECK: %[[WHOLE:.*]] = fopt.view %arg0 : {{.*}} -> !fopt.hbmv<f32, <"2">, <"CHIP{2}, [[R]]{32}, [[C]]{64}">, false>
// CHECK: %[[BLOCK:.*]] = fopt.view %[[X]] : {{.*}} -> !fopt.hbmv<f32, <"CHIP{2}">, <"[[R]]{32}, [[C]]{64}">, true>
// CHECK: fopt.hbm_to_hbm_view %[[WHOLE]], %[[BLOCK]]
// CHECK: %[[OUT:.*]] = fopt.alloc_hbm : !fopt.hbm<f32, <"2">, <"CHIP{2}, [[OR]]{32}, [[OC]]{64}">>
// CHECK: %[[FROM:.*]] = fopt.view %{{.*}} : {{.*}} -> !fopt.hbmv<f32, <"CHIP{2}">, <"[[OR]]{32}, [[OC]]{64}">, false>
// CHECK: %[[TO:.*]] = fopt.view %[[OUT]] : {{.*}} -> !fopt.hbmv<f32, <"2">, <"CHIP{2}, [[OR]]{32}, [[OC]]{64}">, true>
// CHECK: fopt.hbm_to_hbm_view %[[FROM]], %[[TO]]
// CHECK: return %[[OUT]]
// CHECK-LABEL: func.func @accumulate
// CHECK-SAME: (%arg0: !fopt.hbm<f32, <"2">, <"CHIP{2}, {{[A-Z0-9_]+}}{2048}">>
// CHECK-SAME: %arg1: !fopt.hbm<f32, <"2">, <"CHIP{2}, {{[A-Z0-9_]+}}{2048}">>
// CHECK: return
// CHECK-LABEL: func.func @transposed
// CHECK-SAME: (%arg0: !fopt.hbm<f32, <"CHIP{2}">,
// CHECK-SAME: ["CHIP", 2]
// CHECK-SAME: fopt.chip = 2
// CHECK-NOT: {{(hbm|hbmv|dm|tu)<[a-z0-9]+, <"(1|2)">}}
// CHECK: return
// CHECK-LABEL: func.func @pool
// CHECK-SAME: (%arg0: !fopt.hbm<f32, <"2">, <"[[PR:[A-Z0-9_]+]]{8}, CHIP{2}, [[PC:[A-Z0-9_]+]]{32}">>
// CHECK-SAME: -> (!fopt.hbm<f32, <"2">, <"{{[A-Z0-9_]+}}{7}, CHIP{2}, {{[A-Z0-9_]+}}{32}">>
// CHECK: fopt.hbm_to_hbm_view {{.*}} : !fopt.hbmv<f32, <"2">, <"[[PR]]{8}, CHIP{2}, [[PC]]{32}">, false>, !fopt.hbmv<f32, <"CHIP{2}">, <"[[PR]]{8}, [[PC]]{32}">, true>
// CHECK: return

#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d2, d1)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
#tr = affine_map<(d0, d1) -> (d1, d0)>
module attributes {furiosa.chips = 2 : i64} {
  func.func @mm(%x: tensor<64x64xf32>, %w: tensor<64x64xf32>, %out: tensor<64x64xf32>) -> tensor<64x64xf32>
      attributes {furiosa.written = [2]} {
    %zero = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<64x64xf32>
    %acc = linalg.fill ins(%zero : f32) outs(%e : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%x, %w : tensor<64x64xf32>, tensor<64x64xf32>) outs(%acc : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %p = arith.mulf %a, %b : f32
      %s = arith.addf %c, %p : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    %e2 = tensor.empty() : tensor<64x64xf32>
    %relu = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%mm : tensor<64x64xf32>) outs(%e2 : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32):
      %r = arith.maximumf %a, %zero : f32
      linalg.yield %r : f32
    } -> tensor<64x64xf32>
    return %relu : tensor<64x64xf32>
  }

  func.func @accumulate(%x: tensor<64x64xf32>, %out: tensor<64x64xf32>) -> tensor<64x64xf32>
      attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x, %out : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %sum : tensor<64x64xf32>
  }

  func.func @transposed(%x: tensor<64x64xf32>, %out: tensor<64x64xf32>) -> tensor<64x64xf32>
      attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x64xf32>
    %sum = linalg.generic {indexing_maps = [#id, #tr, #id], iterator_types = ["parallel", "parallel"]}
        ins(%x, %out : tensor<64x64xf32>, tensor<64x64xf32>) outs(%e : tensor<64x64xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<64x64xf32>
    return %sum : tensor<64x64xf32>
  }

  func.func @pool(%x: tensor<8x64xf32>, %y: tensor<7x64xf32>) -> tensor<7x64xf32>
      attributes {furiosa.written = [1]} {
    %a = tensor.extract_slice %x[0, 0] [7, 64] [1, 1] : tensor<8x64xf32> to tensor<7x64xf32>
    %b = tensor.extract_slice %x[1, 0] [7, 64] [1, 1] : tensor<8x64xf32> to tensor<7x64xf32>
    %e = tensor.empty() : tensor<7x64xf32>
    %m = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<7x64xf32>, tensor<7x64xf32>) outs(%e : tensor<7x64xf32>) {
    ^bb0(%p: f32, %q: f32, %c: f32):
      %r = arith.maximumf %p, %q : f32
      linalg.yield %r : f32
    } -> tensor<7x64xf32>
    return %m : tensor<7x64xf32>
  }
}
