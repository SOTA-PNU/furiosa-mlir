// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// Attention's P x V divided by the row sums of P over 2048 rows of 512 keys and 64 columns. Merged into one step,
// the tiled class would reduce P on the Vector Engine replicated along the 64 columns its sums are broadcast over:
// each staged read 2048 x 512 x 64 elements, beyond the budget furiosa-opt builds in (its memory grows with them).
// The kernel splits instead: P and its sums are steps of their own, and P x V runs on the Contraction Engine.

// CHECK-LABEL: func.func @pv
// CHECK-COUNT-4: call @pv__s
// CHECK-NOT: call @pv__s
#mk = affine_map<(d0, d1) -> (d0, d1)>
#m2 = affine_map<(d0, d1) -> (d0)>
#mnk_p = affine_map<(d0, d1, d2) -> (d0, d2)>
#mnk_v = affine_map<(d0, d1, d2) -> (d2, d1)>
#mnk_o = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @pv(%s: tensor<2048x512xf32>, %mx: tensor<2048xf32>, %v: tensor<512x64xf32>, %o: tensor<2048x64xf32>) -> tensor<2048x64xf32> attributes {furiosa.written = [3]} {
    %zero = arith.constant 0.0 : f32
    %ep = tensor.empty() : tensor<2048x512xf32>
    %p = linalg.generic {indexing_maps = [#mk, #m2, #mk], iterator_types = ["parallel", "parallel"]} ins(%s, %mx : tensor<2048x512xf32>, tensor<2048xf32>) outs(%ep : tensor<2048x512xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %d = arith.subf %a, %b : f32
      %x = math.exp %d : f32
      linalg.yield %x : f32
    } -> tensor<2048x512xf32>
    %eo = tensor.empty() : tensor<2048x64xf32>
    %fo = linalg.fill ins(%zero : f32) outs(%eo : tensor<2048x64xf32>) -> tensor<2048x64xf32>
    %pv = linalg.generic {indexing_maps = [#mnk_p, #mnk_v, #mnk_o], iterator_types = ["parallel", "parallel", "reduction"]} ins(%p, %v : tensor<2048x512xf32>, tensor<512x64xf32>) outs(%fo : tensor<2048x64xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %m = arith.mulf %a, %b : f32
      %t = arith.addf %acc, %m : f32
      linalg.yield %t : f32
    } -> tensor<2048x64xf32>
    %el = tensor.empty() : tensor<2048xf32>
    %fl = linalg.fill ins(%zero : f32) outs(%el : tensor<2048xf32>) -> tensor<2048xf32>
    %l = linalg.generic {indexing_maps = [#mk, #m2, #m2], iterator_types = ["parallel", "reduction"]} ins(%s, %mx : tensor<2048x512xf32>, tensor<2048xf32>) outs(%fl : tensor<2048xf32>) {
    ^bb0(%a: f32, %b: f32, %acc: f32):
      %d = arith.subf %a, %b : f32
      %x = math.exp %d : f32
      %t = arith.addf %acc, %x : f32
      linalg.yield %t : f32
    } -> tensor<2048xf32>
    %r = linalg.generic {indexing_maps = [#mk, #m2, #mk], iterator_types = ["parallel", "parallel"]} ins(%pv, %l : tensor<2048x64xf32>, tensor<2048xf32>) outs(%eo : tensor<2048x64xf32>) {
    ^bb0(%a: f32, %b: f32, %out: f32):
      %q = arith.divf %a, %b : f32
      linalg.yield %q : f32
    } -> tensor<2048x64xf32>
    return %r : tensor<2048x64xf32>
  }
}
