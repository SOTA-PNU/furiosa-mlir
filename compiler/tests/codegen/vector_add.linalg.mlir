// Hand-written raised linalg for harness/cases/vector_add.py (compiler/README.md contract), driving the
// codegen stages until the raising pass emits it.
module {
  func.func @add_kernel(%x_ptr: tensor<2048xf32> loc("x_ptr"), %y_ptr: tensor<2048xf32> loc("y_ptr"),
                        %out_ptr: tensor<2048xf32> loc("out_ptr")) -> tensor<2048xf32>
      attributes {furiosa.written = [2]} {
    %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                         iterator_types = ["parallel"]}
        ins(%x_ptr, %y_ptr : tensor<2048xf32>, tensor<2048xf32>) outs(%out_ptr : tensor<2048xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %s = arith.addf %a, %b : f32
      linalg.yield %s : f32
    } -> tensor<2048xf32>
    return %r : tensor<2048xf32>
  }
}
