// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A contraction factor computed from w laid out [k, n] (here |w| through its bits, which no class folds into the
// contraction) becomes a generic over [n, k] reading w transposed; that generic reads w relaid out into loop order
// instead, and a padded w (masked K) relays out its source, padded after: a transpose step of the buffer (one DMA),
// the factor reading the pad of it in order, then the contraction.

// CHECK-LABEL: func.func private @dequant__s0
// CHECK: linalg.transpose ins(%arg0 : tensor<40x64xbf16>) outs(%{{.*}} : tensor<64x40xbf16>) permutation = [1, 0]
// CHECK-LABEL: func.func private @dequant__s1
// CHECK: %[[P:.*]] = tensor.pad %arg0 low[0, 0] high[0, 8]
// CHECK: } : tensor<64x40xbf16> to tensor<64x48xbf16>
// CHECK: linalg.generic {indexing_maps = [#[[ID:map[0-9]*]], #[[ID]]], iterator_types = ["parallel", "parallel"]} ins(%[[P]] : tensor<64x48xbf16>)
// CHECK: arith.andi
// CHECK-LABEL: func.func private @dequant__s2
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<32x48xbf16>, tensor<64x48xf32>)
#a = affine_map<(m, n, k) -> (m, k)>
#b = affine_map<(m, n, k) -> (k, n)>
#c = affine_map<(m, n, k) -> (m, n)>
module {
  func.func @dequant(%x: tensor<32x48xbf16>, %w: tensor<40x64xbf16>, %o: tensor<32x64xf32>) -> tensor<32x64xf32> attributes {furiosa.written = [2]} {
    %zb = arith.constant 0.0 : bf16
    %z = arith.constant 0.0 : f32
    %abs = arith.constant 2147483647 : i32
    %wp = tensor.pad %w low[0, 0] high[8, 0] {
    ^bb0(%i: index, %j: index):
      tensor.yield %zb : bf16
    } : tensor<40x64xbf16> to tensor<48x64xbf16>
    %e = tensor.empty() : tensor<32x64xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<32x64xf32>) -> tensor<32x64xf32>
    %r = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %wp : tensor<32x48xbf16>, tensor<48x64xbf16>) outs(%f : tensor<32x64xf32>) {
    ^bb0(%p: bf16, %q: bf16, %acc: f32):
      %pf = arith.extf %p : bf16 to f32
      %qf = arith.extf %q : bf16 to f32
      %qb = arith.bitcast %qf : f32 to i32
      %qa = arith.andi %qb, %abs : i32
      %qs = arith.bitcast %qa : i32 to f32
      %m = arith.mulf %pf, %qs : f32
      %n = arith.addf %acc, %m : f32
      linalg.yield %n : f32
    } -> tensor<32x64xf32>
    return %r : tensor<32x64xf32>
  }
}
