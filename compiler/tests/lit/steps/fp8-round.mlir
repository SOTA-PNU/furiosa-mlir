// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s --implicit-check-not=f8E4M3FN

// A truncation to fp8 rounds in f32 ops the Vector Engine has, as the GPU converts (nearest even, saturating, NaN
// kept), before any kernel maps whole: m = min(|x|, largest), (m + u) - u with u = max(2^exponent(m), least normal)
// * 2^(23 - mantissa bits), then x's sign. The value is exact in fp8 then: its extension is the value itself, and a
// truncation that stays (a store to an fp8 buffer) is exact, so mapping drops it.

// CHECK-LABEL: func.func @quantize
// CHECK: %[[X:.*]] = arith.extf %{{.*}} : bf16 to f32
// CHECK: %[[XB:.*]] = arith.bitcast %[[X]] : f32 to i32
// CHECK: %[[LARGEST:.*]] = arith.constant 4.480000e+02 : f32
// CHECK: %[[ABS:.*]] = arith.andi %[[XB]], %c2147483647_i32 : i32
// CHECK: %[[M:.*]] = arith.minimumf %{{.*}}, %[[LARGEST]] : f32
// CHECK: %[[LEAST:.*]] = arith.constant 1.562500e-02 : f32
// CHECK: %[[MB:.*]] = arith.bitcast %[[M]] : f32 to i32
// CHECK: arith.andi %[[MB]], %c2139095040_i32 : i32
// CHECK: %[[E:.*]] = arith.maximumf %{{.*}}, %[[LEAST]] : f32
// CHECK: %[[SCALE:.*]] = arith.constant 0x49800000 : f32
// CHECK: %[[U:.*]] = arith.mulf %[[E]], %[[SCALE]] : f32
// CHECK: %[[S:.*]] = arith.addf %[[M]], %[[U]] : f32
// CHECK: %[[R:.*]] = arith.subf %[[S]], %[[U]] : f32
// CHECK: %[[SIGN:.*]] = arith.andi %[[XB]], %c-2147483648_i32 : i32
// CHECK: %[[RB:.*]] = arith.bitcast %[[R]] : f32 to i32
// CHECK: %[[V:.*]] = arith.ori %[[RB]], %[[SIGN]] : i32
// CHECK: %[[VF:.*]] = arith.bitcast %[[V]] : i32 to f32
// CHECK: linalg.yield %[[VF]] : f32
// e5m2: largest 57344, least normal 2^-14, u = 2^(exponent + 21); the store's truncation stays.
// CHECK-LABEL: func.func @store
// CHECK: arith.constant 5.734400e+04 : f32
// CHECK: arith.constant 6.10351563E-5 : f32
// CHECK: arith.constant 0x4A000000 : f32
// CHECK: %[[T:.*]] = arith.truncf %{{.*}} : f32 to f8E5M2
// CHECK: linalg.yield %[[T]] : f8E5M2
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @quantize(%x: tensor<64x256xbf16>, %o: tensor<64x256xf32>) -> tensor<64x256xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x256xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x256xbf16>) outs(%e : tensor<64x256xf32>) {
    ^bb0(%a: bf16, %out: f32):
      %q = arith.truncf %a : bf16 to f8E4M3FN
      %w = arith.extf %q : f8E4M3FN to f32
      linalg.yield %w : f32
    } -> tensor<64x256xf32>
    return %r : tensor<64x256xf32>
  }
  func.func @store(%x: tensor<64x256xf32>, %o: tensor<64x256xf8E5M2>) -> tensor<64x256xf8E5M2> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64x256xf8E5M2>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<64x256xf32>) outs(%e : tensor<64x256xf8E5M2>) {
    ^bb0(%a: f32, %out: f8E5M2):
      %q = arith.truncf %a : f32 to f8E5M2
      linalg.yield %q : f8E5M2
    } -> tensor<64x256xf8E5M2>
    return %r : tensor<64x256xf8E5M2>
  }
}
