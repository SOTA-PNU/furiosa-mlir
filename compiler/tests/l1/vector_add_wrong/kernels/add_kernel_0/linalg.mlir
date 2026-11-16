// Deliberately wrong (subtracts): the negative check for L1. Hand-written raised form of harness/cases/vector_add.py's add_kernel (compiler/README.md contract):
// the grid of 4 programs x 512 lanes folded into one elementwise op over the whole buffer.
#map = affine_map<(d0) -> (d0)>
module {
  func.func public @add_kernel(%x: tensor<2048xf32>, %y: tensor<2048xf32>, %out: tensor<2048xf32>) -> tensor<2048xf32>
      attributes {furiosa.written = [2]} {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
        ins(%x, %y : tensor<2048xf32>, tensor<2048xf32>) outs(%out : tensor<2048xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %s = arith.subf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<2048xf32>
    return %0 : tensor<2048xf32>
  }
}
