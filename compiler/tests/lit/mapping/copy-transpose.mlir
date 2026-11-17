// RUN: furiosa-mlir-opt --furiosa-linalg-to-fopt %s | FileCheck %s

// The copy class maps a linalg.transpose of a buffer (attention's V^T for the PV contraction) as one HBM-to-HBM
// DMA between the two layouts: the axes name the elements, the maps lay them out.

// CHECK-LABEL: func.func @vt
// CHECK: %[[SRC:.*]] = fopt.view %arg0 : !fopt.hbm<bf16, <"1">, <"D0{64}, D1{128}">> -> !fopt.hbmv<bf16, <"1">, <"D0{64}, D1{128}">, false>
// CHECK: %[[DST:.*]] = fopt.view %arg1 : !fopt.hbm<bf16, <"1">, <"D1{128}, D0{64}">> -> !fopt.hbmv<bf16, <"1">, <"D1{128}, D0{64}">, true>
// CHECK: fopt.hbm_to_hbm_view %[[SRC]], %[[DST]]
module {
  func.func @vt(%v: tensor<64x128xbf16>, %o: tensor<128x64xbf16>) -> tensor<128x64xbf16> attributes {furiosa.written = [1]} {
    %e = tensor.empty() : tensor<128x64xbf16>
    %t = linalg.transpose ins(%v : tensor<64x128xbf16>) outs(%e : tensor<128x64xbf16>) permutation = [1, 0]
    return %t : tensor<128x64xbf16>
  }
}
