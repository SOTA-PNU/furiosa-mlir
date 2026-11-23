// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// The gather class maps an embedding lookup, out[t, :] = emb[ids[t], :]: the token ids become byte offsets of
// whole rows (id * row bytes, on the Vector Engine), and one gather DMA reads those rows of the table. An index list
// that would not fit one slice's DM with its offsets (65536 ids) spreads over the slices as the rows do.

// CHECK-LABEL: func.func @embed
// CHECK: fopt.vector_fxp "MulInt" %{{.*}}, 128 :
// CHECK: fopt.gather %arg1[%{{.*}}] indexed <"V{1024}"> payload <"S{64}">

// CHECK-LABEL: func.func @embed_long
// CHECK: fopt.to_dm %arg0 : {{.*}} -> !fopt.dm<i32, <"1">, <"1 # 2">, <"T{65536} / 256">, <"T{65536} % 256">>
// CHECK: fopt.vector_fxp "MulInt" %{{.*}}, 8 :
// CHECK: fopt.gather %arg1[%{{.*}}] indexed <"V{1024}"> payload <"S{4}">
#ids = affine_map<(d0, d1) -> (d0)>
#out = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @embed(%ids: tensor<8xi32>, %emb: tensor<1024x64xbf16>, %o: tensor<8x64xbf16>) -> tensor<8x64xbf16> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<8x64xbf16>
    %r = linalg.generic {indexing_maps = [#ids, #out], iterator_types = ["parallel", "parallel"]} ins(%ids : tensor<8xi32>) outs(%e : tensor<8x64xbf16>) {
    ^bb0(%id: i32, %out: bf16):
      %h = linalg.index 1 : index
      %row = arith.index_cast %id : i32 to index
      %v = tensor.extract %emb[%row, %h] : tensor<1024x64xbf16>
      linalg.yield %v : bf16
    } -> tensor<8x64xbf16>
    return %r : tensor<8x64xbf16>
  }
  func.func @embed_long(%ids: tensor<65536xi32>, %emb: tensor<1024x4xbf16>, %o: tensor<65536x4xbf16>) -> tensor<65536x4xbf16> attributes {furiosa.written = [2]} {
    %e = tensor.empty() : tensor<65536x4xbf16>
    %r = linalg.generic {indexing_maps = [#ids, #out], iterator_types = ["parallel", "parallel"]} ins(%ids : tensor<65536xi32>) outs(%e : tensor<65536x4xbf16>) {
    ^bb0(%id: i32, %out: bf16):
      %h = linalg.index 1 : index
      %row = arith.index_cast %id : i32 to index
      %v = tensor.extract %emb[%row, %h] : tensor<1024x4xbf16>
      linalg.yield %v : bf16
    } -> tensor<65536x4xbf16>
    return %r : tensor<65536x4xbf16>
  }
}
