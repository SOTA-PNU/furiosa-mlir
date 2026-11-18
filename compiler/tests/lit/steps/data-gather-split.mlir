// RUN: furiosa-mlir-opt --furiosa-split-steps %s | FileCheck %s

// Cross-entropy's target logit x[r, t[r]] is a data-dependent read inside the loss computation. The read moves
// into a generic of its own over the loops its address uses (here the row), which gathers as an element gather
// does (8-byte rows of x seen as [192, 4] bf16, a select by the lane), and the loss reads the gathered values as
// an input. The gather DMA takes a power of two of rows: the 12 row addresses pad to 16 (row 0) and the lanes keep
// the first 12. A read at one data-dependent address (@pick1: x[t[0]] in a one-element generic) gathers over one
// loop of one iteration, along which both the address generic and the consumer read their one-entry operands.

// CHECK-DAG: #[[ID:map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func private @loss__s1(%arg0: tensor<12xi32>
// CHECK: tensor.pad %arg0 low[0] high[4]
// CHECK: arith.shrsi %{{.*}}, %c2_i32
// CHECK-LABEL: func.func private @loss__s2
// CHECK: tensor.extract %{{.*}} : tensor<192x4xbf16>
// CHECK: } -> tensor<16x4xbf16>
// CHECK-LABEL: func.func private @loss__s3
// CHECK: tensor.extract_slice %arg0[0, 0] [12, 1] [1, 1] : tensor<16x4xbf16> to tensor<12x1xbf16>
// CHECK-LABEL: func.func private @loss__s7
// CHECK-NOT: tensor.extract
// CHECK: arith.select %{{.*}} : bf16
// CHECK: %[[X:.*]] = arith.extf %{{.*}} : bf16 to f32
// CHECK: arith.subf %{{.*}}, %[[X]] : f32
// CHECK-LABEL: func.func private @pick1__s0(%arg0: tensor<1xi32>
// CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[ID]]]
// CHECK-LABEL: func.func private @pick1__s6
// CHECK: linalg.generic {indexing_maps = [#{{map[0-9]*}}, #[[ID]], #[[ID]]], {{.*}} ins(%arg5, %{{.*}} : tensor<f32>, tensor<1xbf16>)
#id = affine_map<(d0) -> (d0)>
#s = affine_map<(d0) -> ()>
#z = affine_map<(d0) -> (0)>
module {
  func.func @loss(%logits: tensor<12x64xbf16>, %lse: tensor<12xf32>, %target: tensor<12xi32>, %o: tensor<12xf32>) -> tensor<12xf32> attributes {furiosa.written = [3]} {
    %c767 = arith.constant 767 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %flat = tensor.collapse_shape %logits [[0, 1]] : tensor<12x64xbf16> into tensor<768xbf16>
    %e = tensor.empty() : tensor<12xf32>
    %r = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel"]} ins(%lse, %target : tensor<12xf32>, tensor<12xi32>) outs(%e : tensor<12xf32>) {
    ^bb0(%l: f32, %t: i32, %out: f32):
      %i = linalg.index 0 : index
      %row = arith.muli %i, %c64 : index
      %col = arith.index_cast %t : i32 to index
      %a = arith.addi %row, %col : index
      %lo = arith.maxsi %a, %c0 : index
      %hi = arith.minsi %lo, %c767 : index
      %x = tensor.extract %flat[%hi] : tensor<768xbf16>
      %xf = arith.extf %x : bf16 to f32
      %d = arith.subf %l, %xf : f32
      linalg.yield %d : f32
    } -> tensor<12xf32>
    return %r : tensor<12xf32>
  }
  func.func @pick1(%logits: tensor<12x64xbf16>, %lse: tensor<f32>, %target: tensor<1xi32>, %o: tensor<1xf32>) -> tensor<1xf32> attributes {furiosa.written = [3]} {
    %c767 = arith.constant 767 : index
    %c0 = arith.constant 0 : index
    %flat = tensor.collapse_shape %logits [[0, 1]] : tensor<12x64xbf16> into tensor<768xbf16>
    %e = tensor.empty() : tensor<1xf32>
    %r = linalg.generic {indexing_maps = [#s, #z, #id], iterator_types = ["parallel"]} ins(%lse, %target : tensor<f32>, tensor<1xi32>) outs(%e : tensor<1xf32>) {
    ^bb0(%l: f32, %t: i32, %out: f32):
      %col = arith.index_cast %t : i32 to index
      %lo = arith.maxsi %col, %c0 : index
      %hi = arith.minsi %lo, %c767 : index
      %x = tensor.extract %flat[%hi] : tensor<768xbf16>
      %xf = arith.extf %x : bf16 to f32
      %d = arith.subf %l, %xf : f32
      linalg.yield %d : f32
    } -> tensor<1xf32>
    return %r : tensor<1xf32>
  }
}
