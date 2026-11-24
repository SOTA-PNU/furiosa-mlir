// RUN: furiosa-mlir-opt --furiosa-tile --furiosa-linalg-to-fopt %s | FileCheck %s

// exp(x . w^T) at 1500 x 1536 (K 32) takes the 1500 as its column: 4 per slice block (the largest power of two
// dividing 1500), in packets padded to 8, over column tiles of 256 blocks that pad it to 2048 (CP). The result's
// relayout reads the canonical layout back as the column padded to 2048, its local 4 columns still padded to the
// packet, so the reshape keeps the size.

// CHECK-LABEL: func.func @mm
// CHECK: fopt.reshape_hbm %{{.*}} : !fopt.hbm<f32, <"1">, <"A1{1536}, CP{2048} / 1024, CP{2048} / 4 % 256, CP{2048} % 4 # 8">> -> !fopt.hbm<f32, <"1">, <"A1{1536}, A0{1500} # 2048 / 4, A0{1500} # 2048 % 4 # 8">>
#x = affine_map<(d0, d1, d2) -> (d0, d2)>
#w = affine_map<(d0, d1, d2) -> (d1, d2)>
#o = affine_map<(d0, d1, d2) -> (d0, d1)>
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @mm(%x: tensor<1500x32xf32>, %w: tensor<1536x32xf32>, %o: tensor<1500x1536xf32>) -> tensor<1500x1536xf32> attributes {furiosa.written = [2]} {
    %z = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<1500x1536xf32>
    %f = linalg.fill ins(%z : f32) outs(%e : tensor<1500x1536xf32>) -> tensor<1500x1536xf32>
    %r = linalg.generic {indexing_maps = [#x, #w, #o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x, %w : tensor<1500x32xf32>, tensor<1536x32xf32>) outs(%f : tensor<1500x1536xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %m = arith.mulf %a, %b : f32
      %s = arith.addf %out, %m : f32
      linalg.yield %s : f32
    } -> tensor<1500x1536xf32>
    %g = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]} ins(%r : tensor<1500x1536xf32>) outs(%e : tensor<1500x1536xf32>) {
    ^bb0(%a: f32, %out: f32):
      %t = math.exp %a : f32
      linalg.yield %t : f32
    } -> tensor<1500x1536xf32>
    return %g : tensor<1500x1536xf32>
  }
}
