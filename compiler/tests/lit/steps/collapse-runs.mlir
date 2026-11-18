// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// A generic over twelve loops of extent 2 (tl.sort's stages) that no class maps: the row-wise class splits its
// outermost row dim over the 256 slices, and here the rows inside it are 1024. Loops 0 to 9, which every operand reads
// together and in order, become one loop of 1024 (the operands and result reshaped): the row-wise class maps that
// generic once it is a step of its own (reading the reshaped buffers).

// CHECK-LABEL: func.func @bits
// CHECK: tensor.collapse_shape %arg0 {{\[}}[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10], [11]] : {{.*}} into tensor<1024x2x2xi32>
// CHECK: tensor.collapse_shape %arg1 {{\[}}[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]] : {{.*}} into tensor<1024x2xi32>
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel"]
#id = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11)>
#partner = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d11)>
module {
  func.func @bits(%x: tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>, %p: tensor<2x2x2x2x2x2x2x2x2x2x2xi32>, %o: tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>
    %r = linalg.generic {indexing_maps = [#id, #partner, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%x, %p : tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>, tensor<2x2x2x2x2x2x2x2x2x2x2xi32>) outs(%e : tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %s = arith.xori %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>
    return %r : tensor<2x2x2x2x2x2x2x2x2x2x2x2xi32>
  }
}
