// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=math.fma --implicit-check-not=math.log2 --implicit-check-not=math.floor --implicit-check-not=math.asin --implicit-check-not=math.acos --implicit-check-not=math.atan --implicit-check-not=math.copysign --implicit-check-not=arith.minui --implicit-check-not=arith.mului_extended

// Math ops the Vector Engine lacks become ones it has: fma(a, b, c) = a*b + c, log2(x) = ln(x) * log2(e), and
// floor(x) = t - (t > x) from the nearest integer t, x itself beyond 2^23 (already an integer); asin, acos,
// atan and atan2 by MLIR's polynomial approximations, copysign(a, b) = b < 0 ? -|a| : |a|; an unsigned min compares
// signed with the sign bits flipped; tl.umulhi's high half comes from 16-bit limbs (four wrapping i32 products).

// CHECK-LABEL: func.func @fma_log2
// CHECK: ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[M:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK: %[[S:.*]] = arith.addf %[[M]], %[[C]] : f32
// CHECK-DAG: %[[E:.*]] = arith.constant 1.44269502 : f32
// CHECK-DAG: %[[L:.*]] = math.log %[[S]] : f32
// CHECK: arith.mulf %[[L]], %[[E]] : f32
// CHECK-LABEL: func.func @floor
// CHECK: ^bb0(%[[X:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[T:.*]] = math.round %[[X]] : f32
// CHECK: %[[PAST:.*]] = arith.cmpf ogt, %[[T]], %[[X]] : f32
// CHECK: %[[STEP:.*]] = arith.select %[[PAST]]
// CHECK: %[[NEAR:.*]] = arith.subf %[[T]], %[[STEP]] : f32
// CHECK-DAG: arith.cmpf olt, %[[X]], %{{.*}} : f32
// CHECK-DAG: arith.cmpf ogt, %[[X]], %{{.*}} : f32
// CHECK: arith.select %{{.*}}, %[[X]], %[[NEAR]] : f32
// CHECK: arith.select %{{.*}}, %[[X]], %{{.*}} : f32
// CHECK-LABEL: func.func @inverse_trig
// CHECK: math.sqrt
// CHECK: math.absf
// CHECK-LABEL: func.func @unsigned_min
// CHECK: %[[S:.*]] = arith.constant -2147483648 : i32
// CHECK: %[[A:.*]] = arith.xori %{{.*}}, %[[S]] : i32
// CHECK: %[[B:.*]] = arith.xori %{{.*}}, %[[S]] : i32
// CHECK: %[[M:.*]] = arith.minsi %[[A]], %[[B]] : i32
// CHECK: arith.xori %[[M]], %[[S]] : i32
// CHECK-LABEL: func.func @mulhi
// CHECK-COUNT-4: arith.muli
// CHECK: arith.shrui
#id = affine_map<(d0) -> (d0)>
module {
  func.func @fma_log2(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> attributes {furiosa.written = [3]} {
    %e = tensor.empty() : tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id, #id], iterator_types = ["parallel"]} ins(%a, %b, %c : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32, %out: f32):
      %f = math.fma %x, %y, %z : f32
      %l = math.log2 %f : f32
      linalg.yield %l : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
  func.func @floor(%a: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%a : tensor<64xf32>) outs(%e : tensor<64xf32>) {
    ^bb0(%x: f32, %out: f32):
      %f = math.floor %x : f32
      linalg.yield %f : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
  func.func @inverse_trig(%a: tensor<64xf32>, %b: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%a, %b : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xf32>) {
    ^bb0(%x: f32, %y: f32, %out: f32):
      %s = math.asin %x : f32
      %c = math.acos %x : f32
      %t = math.atan %x : f32
      %u = math.atan2 %x, %y : f32
      %0 = arith.addf %s, %c : f32
      %1 = arith.addf %t, %u : f32
      %2 = arith.addf %0, %1 : f32
      linalg.yield %2 : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
  func.func @unsigned_min(%a: tensor<64xi32>, %b: tensor<64xi32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%a, %b : tensor<64xi32>, tensor<64xi32>) outs(%e : tensor<64xi32>) {
    ^bb0(%x: i32, %y: i32, %out: i32):
      %m = arith.minui %x, %y : i32
      linalg.yield %m : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
  func.func @mulhi(%a: tensor<64xi32>, %b: tensor<64xi32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%a, %b : tensor<64xi32>, tensor<64xi32>) outs(%e : tensor<64xi32>) {
    ^bb0(%x: i32, %y: i32, %out: i32):
      %l, %h = arith.mului_extended %x, %y : i32
      linalg.yield %h : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}
