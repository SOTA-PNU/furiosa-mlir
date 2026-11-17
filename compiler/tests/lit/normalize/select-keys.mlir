// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=arith.andi --implicit-check-not=arith.xori

// tl.map_elementwise's three-way compare: selects keyed by combinations of comparisons, (!lt & eq) and
// (!lt & !eq), become nested selects each keyed by one comparison (a negation swaps the select's values).

// CHECK-LABEL: func.func @three_way
// CHECK-DAG: %[[LT:.*]] = arith.cmpf olt, %[[X:.*]], %[[Y:.*]] : f32
// CHECK-DAG: %[[EQ:.*]] = arith.cmpf oeq, %[[X]], %[[Y]] : f32
// CHECK-DAG: arith.select %[[EQ]], %{{.*}}, %{{.*}} : i32
// CHECK-DAG: arith.select %[[LT]], %{{.*}}, %{{.*}} : i32
#id = affine_map<(d0) -> (d0)>
module {
  func.func @three_way(%a: tensor<64xf32>, %b: tensor<64xf32>, %o: tensor<64xi32>) -> tensor<64xi32> attributes {furiosa.written = [2]} {
    %true = arith.constant true
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cm1 = arith.constant -1 : i32
    %e = tensor.empty() : tensor<64xi32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%a, %b : tensor<64xf32>, tensor<64xf32>) outs(%e : tensor<64xi32>) {
    ^bb0(%x: f32, %y: f32, %out: i32):
      %lt = arith.cmpf olt, %x, %y : f32
      %ge = arith.xori %lt, %true : i1
      %eq = arith.cmpf oeq, %x, %y : f32
      %ne = arith.xori %eq, %true : i1
      %same = arith.andi %ge, %eq : i1
      %more = arith.andi %ge, %ne : i1
      %s = arith.select %same, %c0, %cm1 : i32
      %t = arith.select %more, %c1, %s : i32
      linalg.yield %t : i32
    } -> tensor<64xi32>
    return %r : tensor<64xi32>
  }
}
