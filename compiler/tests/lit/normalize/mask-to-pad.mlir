// RUN: furiosa-mlir-opt --furiosa-normalize-kernels %s | FileCheck %s --implicit-check-not=arith.select

// A masked read, select(n < 33, k[n], -inf), the mask an index-only bound on a loop dim the input spans and
// true on one interval of it, becomes a read of k's box on that interval padded with the select's constant:
// the box inserted into a fill (tensor.pad's meaning, in ops the copy steps map).

// CHECK-LABEL: func.func @masked
// CHECK: %[[BOX:.*]] = tensor.extract_slice %arg0[0] [33] [1] : tensor<64xf32> to tensor<33xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<64xf32>)
// CHECK: %[[PAD:.*]] = tensor.insert_slice %[[BOX]] into %[[FILL]][0] [33] [1]
// CHECK: linalg.generic {{.*}} ins(%[[PAD]] : tensor<64xf32>)
#id = affine_map<(d0) -> (d0)>
module {
  func.func @masked(%k: tensor<64xf32>, %o: tensor<64xf32>) -> tensor<64xf32> attributes {furiosa.written = [1]} {
    %ninf = arith.constant 0xFF800000 : f32
    %c33 = arith.constant 33 : index
    %e = tensor.empty() : tensor<64xf32>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]} ins(%k : tensor<64xf32>) outs(%e : tensor<64xf32>) {
    ^bb0(%a: f32, %out: f32):
      %n = linalg.index 0 : index
      %in = arith.cmpi slt, %n, %c33 : index
      %v = arith.select %in, %a, %ninf : f32
      %s = math.exp %v : f32
      linalg.yield %s : f32
    } -> tensor<64xf32>
    return %r : tensor<64xf32>
  }
}
