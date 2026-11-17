// RUN: furiosa-mlir-opt --furiosa-raise-to-linalg %s | FileCheck %s --implicit-check-not=tt.reduce

// tl.sort on 2x16 rows: Triton reshapes each row to 2x2x2x2 and runs a bitonic network of xor
// reductions in that space, each stage reading the previous one directly and through a reduction.
// Gather form would build every reduction once per context that reads it (2047 generics here);
// built once each in their own index space, the 10 reductions take two generics apiece.

// CHECK-LABEL: func.func @scan_kernel
// CHECK-COUNT-10: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]
// CHECK-NOT: "reduction"
// CHECK: return

// Triton source (harness/ops scan_kernel, OP=3): y = tl.sort(x, dim=1) on a 2x16 block.

module attributes {furiosa.launch = "{\22grid\22: [1, 1, 1], \22buffers\22: [{\22arg\22: 0, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 16]}, {\22arg\22: 1, \22kind\22: \22state\22, \22dtype\22: \22fp32\22, \22shape\22: [2, 16]}]}"} {
  tt.func public @scan_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<1x1x1x2x2xi32>
    %cst_0 = arith.constant dense<0> : tensor<1x1x2x1x2xi32>
    %cst_1 = arith.constant dense<0> : tensor<1x1x2x2x1xi32>
    %cst_2 = arith.constant dense<0> : tensor<1x2x1x1x2xi32>
    %cst_3 = arith.constant dense<0> : tensor<1x2x1x2x1xi32>
    %cst_4 = arith.constant dense<0> : tensor<1x2x2x1x1xi32>
    %cst_5 = arith.constant dense<0> : tensor<1x1x1x1x2xi32>
    %cst_6 = arith.constant dense<0> : tensor<1x1x1x2x1xi32>
    %cst_7 = arith.constant dense<0> : tensor<1x1x2x1x1xi32>
    %cst_8 = arith.constant dense<0> : tensor<1x2x1x1x1xi32>
    %cst_9 = arith.constant dense<0.000000e+00> : tensor<2x16xf32>
    %cst_10 = arith.constant dense<16> : tensor<2x1xi32>
    %cst_11 = arith.constant dense<2> : tensor<2x1xi32>
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %4 = arith.cmpi slt, %1, %cst_11 : tensor<2x1xi32>
    %5 = arith.muli %1, %cst_10 : tensor<2x1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %7 = tt.addptr %6, %5 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %8 = tt.broadcast %7 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x16x!tt.ptr<f32>>
    %9 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<2x16xi32>
    %10 = tt.addptr %8, %9 : tensor<2x16x!tt.ptr<f32>>, tensor<2x16xi32>
    %11 = tt.broadcast %4 : tensor<2x1xi1> -> tensor<2x16xi1>
    %12 = tt.load %10, %11, %cst_9 : tensor<2x16x!tt.ptr<f32>>
    %13 = tt.reshape %12 : tensor<2x16xf32> -> tensor<2x2x2x2x2xf32>
    %14 = tt.reshape %0 : tensor<2xi32> -> tensor<1x1x1x2x1xi32>
    %15 = tt.bitcast %13 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %16 = "tt.reduce"(%15) <{axis = 4 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %17 = tt.expand_dims %16 {axis = 4 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x2x1xi32>
    %18 = tt.broadcast %17 : tensor<2x2x2x2x1xi32> -> tensor<2x2x2x2x2xi32>
    %19 = arith.xori %15, %18 : tensor<2x2x2x2x2xi32>
    %20 = tt.bitcast %19 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %21 = tt.reshape %0 : tensor<2xi32> -> tensor<1x1x1x1x2xi32>
    %22 = arith.cmpf ogt, %13, %20 : tensor<2x2x2x2x2xf32>
    %23 = tt.broadcast %14 : tensor<1x1x1x2x1xi32> -> tensor<1x1x1x2x2xi32>
    %24 = tt.broadcast %21 : tensor<1x1x1x1x2xi32> -> tensor<1x1x1x2x2xi32>
    %25 = arith.xori %23, %24 : tensor<1x1x1x2x2xi32>
    %26 = arith.cmpi ne, %25, %cst : tensor<1x1x1x2x2xi32>
    %27 = tt.broadcast %26 : tensor<1x1x1x2x2xi1> -> tensor<2x2x2x2x2xi1>
    %28 = arith.cmpi ne, %22, %27 : tensor<2x2x2x2x2xi1>
    %29 = arith.select %28, %20, %13 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %30 = tt.reshape %0 : tensor<2xi32> -> tensor<1x1x2x1x1xi32>
    %31 = tt.bitcast %29 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %32 = "tt.reduce"(%31) <{axis = 3 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %33 = tt.expand_dims %32 {axis = 3 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x1x2xi32>
    %34 = tt.broadcast %33 : tensor<2x2x2x1x2xi32> -> tensor<2x2x2x2x2xi32>
    %35 = arith.xori %31, %34 : tensor<2x2x2x2x2xi32>
    %36 = tt.bitcast %35 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %37 = arith.cmpf ogt, %29, %36 : tensor<2x2x2x2x2xf32>
    %38 = tt.broadcast %30 : tensor<1x1x2x1x1xi32> -> tensor<1x1x2x2x1xi32>
    %39 = tt.broadcast %14 : tensor<1x1x1x2x1xi32> -> tensor<1x1x2x2x1xi32>
    %40 = arith.xori %38, %39 : tensor<1x1x2x2x1xi32>
    %41 = arith.cmpi ne, %40, %cst_1 : tensor<1x1x2x2x1xi32>
    %42 = tt.broadcast %41 : tensor<1x1x2x2x1xi1> -> tensor<2x2x2x2x2xi1>
    %43 = arith.cmpi ne, %37, %42 : tensor<2x2x2x2x2xi1>
    %44 = arith.select %43, %36, %29 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %45 = tt.bitcast %44 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %46 = "tt.reduce"(%45) <{axis = 4 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %47 = tt.expand_dims %46 {axis = 4 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x2x1xi32>
    %48 = tt.broadcast %47 : tensor<2x2x2x2x1xi32> -> tensor<2x2x2x2x2xi32>
    %49 = arith.xori %45, %48 : tensor<2x2x2x2x2xi32>
    %50 = tt.bitcast %49 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %51 = arith.cmpf ogt, %44, %50 : tensor<2x2x2x2x2xf32>
    %52 = tt.broadcast %30 : tensor<1x1x2x1x1xi32> -> tensor<1x1x2x1x2xi32>
    %53 = tt.broadcast %21 : tensor<1x1x1x1x2xi32> -> tensor<1x1x2x1x2xi32>
    %54 = arith.xori %52, %53 : tensor<1x1x2x1x2xi32>
    %55 = arith.cmpi ne, %54, %cst_0 : tensor<1x1x2x1x2xi32>
    %56 = tt.broadcast %55 : tensor<1x1x2x1x2xi1> -> tensor<2x2x2x2x2xi1>
    %57 = arith.cmpi ne, %51, %56 : tensor<2x2x2x2x2xi1>
    %58 = arith.select %57, %50, %44 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %59 = tt.reshape %0 : tensor<2xi32> -> tensor<1x2x1x1x1xi32>
    %60 = tt.bitcast %58 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %61 = "tt.reduce"(%60) <{axis = 2 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %62 = tt.expand_dims %61 {axis = 2 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x1x2x2xi32>
    %63 = tt.broadcast %62 : tensor<2x2x1x2x2xi32> -> tensor<2x2x2x2x2xi32>
    %64 = arith.xori %60, %63 : tensor<2x2x2x2x2xi32>
    %65 = tt.bitcast %64 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %66 = arith.cmpf ogt, %58, %65 : tensor<2x2x2x2x2xf32>
    %67 = tt.broadcast %59 : tensor<1x2x1x1x1xi32> -> tensor<1x2x2x1x1xi32>
    %68 = tt.broadcast %30 : tensor<1x1x2x1x1xi32> -> tensor<1x2x2x1x1xi32>
    %69 = arith.xori %67, %68 : tensor<1x2x2x1x1xi32>
    %70 = arith.cmpi ne, %69, %cst_4 : tensor<1x2x2x1x1xi32>
    %71 = tt.broadcast %70 : tensor<1x2x2x1x1xi1> -> tensor<2x2x2x2x2xi1>
    %72 = arith.cmpi ne, %66, %71 : tensor<2x2x2x2x2xi1>
    %73 = arith.select %72, %65, %58 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %74 = tt.bitcast %73 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %75 = "tt.reduce"(%74) <{axis = 3 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %76 = tt.expand_dims %75 {axis = 3 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x1x2xi32>
    %77 = tt.broadcast %76 : tensor<2x2x2x1x2xi32> -> tensor<2x2x2x2x2xi32>
    %78 = arith.xori %74, %77 : tensor<2x2x2x2x2xi32>
    %79 = tt.bitcast %78 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %80 = arith.cmpf ogt, %73, %79 : tensor<2x2x2x2x2xf32>
    %81 = tt.broadcast %59 : tensor<1x2x1x1x1xi32> -> tensor<1x2x1x2x1xi32>
    %82 = tt.broadcast %14 : tensor<1x1x1x2x1xi32> -> tensor<1x2x1x2x1xi32>
    %83 = arith.xori %81, %82 : tensor<1x2x1x2x1xi32>
    %84 = arith.cmpi ne, %83, %cst_3 : tensor<1x2x1x2x1xi32>
    %85 = tt.broadcast %84 : tensor<1x2x1x2x1xi1> -> tensor<2x2x2x2x2xi1>
    %86 = arith.cmpi ne, %80, %85 : tensor<2x2x2x2x2xi1>
    %87 = arith.select %86, %79, %73 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %88 = tt.bitcast %87 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %89 = "tt.reduce"(%88) <{axis = 4 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %90 = tt.expand_dims %89 {axis = 4 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x2x1xi32>
    %91 = tt.broadcast %90 : tensor<2x2x2x2x1xi32> -> tensor<2x2x2x2x2xi32>
    %92 = arith.xori %88, %91 : tensor<2x2x2x2x2xi32>
    %93 = tt.bitcast %92 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %94 = arith.cmpf ogt, %87, %93 : tensor<2x2x2x2x2xf32>
    %95 = tt.broadcast %59 : tensor<1x2x1x1x1xi32> -> tensor<1x2x1x1x2xi32>
    %96 = tt.broadcast %21 : tensor<1x1x1x1x2xi32> -> tensor<1x2x1x1x2xi32>
    %97 = arith.xori %95, %96 : tensor<1x2x1x1x2xi32>
    %98 = arith.cmpi ne, %97, %cst_2 : tensor<1x2x1x1x2xi32>
    %99 = tt.broadcast %98 : tensor<1x2x1x1x2xi1> -> tensor<2x2x2x2x2xi1>
    %100 = arith.cmpi ne, %94, %99 : tensor<2x2x2x2x2xi1>
    %101 = arith.select %100, %93, %87 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %102 = tt.bitcast %101 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %103 = "tt.reduce"(%102) <{axis = 1 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %104 = tt.expand_dims %103 {axis = 1 : i32} : tensor<2x2x2x2xi32> -> tensor<2x1x2x2x2xi32>
    %105 = tt.broadcast %104 : tensor<2x1x2x2x2xi32> -> tensor<2x2x2x2x2xi32>
    %106 = arith.xori %102, %105 : tensor<2x2x2x2x2xi32>
    %107 = tt.bitcast %106 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %108 = arith.cmpf ogt, %101, %107 : tensor<2x2x2x2x2xf32>
    %109 = arith.cmpi ne, %59, %cst_8 : tensor<1x2x1x1x1xi32>
    %110 = tt.broadcast %109 : tensor<1x2x1x1x1xi1> -> tensor<2x2x2x2x2xi1>
    %111 = arith.cmpi ne, %108, %110 : tensor<2x2x2x2x2xi1>
    %112 = arith.select %111, %107, %101 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %113 = tt.bitcast %112 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %114 = "tt.reduce"(%113) <{axis = 2 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %115 = tt.expand_dims %114 {axis = 2 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x1x2x2xi32>
    %116 = tt.broadcast %115 : tensor<2x2x1x2x2xi32> -> tensor<2x2x2x2x2xi32>
    %117 = arith.xori %113, %116 : tensor<2x2x2x2x2xi32>
    %118 = tt.bitcast %117 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %119 = arith.cmpf ogt, %112, %118 : tensor<2x2x2x2x2xf32>
    %120 = arith.cmpi ne, %30, %cst_7 : tensor<1x1x2x1x1xi32>
    %121 = tt.broadcast %120 : tensor<1x1x2x1x1xi1> -> tensor<2x2x2x2x2xi1>
    %122 = arith.cmpi ne, %119, %121 : tensor<2x2x2x2x2xi1>
    %123 = arith.select %122, %118, %112 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %124 = tt.bitcast %123 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %125 = "tt.reduce"(%124) <{axis = 3 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %126 = tt.expand_dims %125 {axis = 3 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x1x2xi32>
    %127 = tt.broadcast %126 : tensor<2x2x2x1x2xi32> -> tensor<2x2x2x2x2xi32>
    %128 = arith.xori %124, %127 : tensor<2x2x2x2x2xi32>
    %129 = tt.bitcast %128 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %130 = arith.cmpf ogt, %123, %129 : tensor<2x2x2x2x2xf32>
    %131 = arith.cmpi ne, %14, %cst_6 : tensor<1x1x1x2x1xi32>
    %132 = tt.broadcast %131 : tensor<1x1x1x2x1xi1> -> tensor<2x2x2x2x2xi1>
    %133 = arith.cmpi ne, %130, %132 : tensor<2x2x2x2x2xi1>
    %134 = arith.select %133, %129, %123 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %135 = tt.bitcast %134 : tensor<2x2x2x2x2xf32> -> tensor<2x2x2x2x2xi32>
    %136 = "tt.reduce"(%135) <{axis = 4 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32):
      %151 = arith.xori %arg2, %arg3 : i32
      tt.reduce.return %151 : i32
    }) : (tensor<2x2x2x2x2xi32>) -> tensor<2x2x2x2xi32>
    %137 = tt.expand_dims %136 {axis = 4 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x2x1xi32>
    %138 = tt.broadcast %137 : tensor<2x2x2x2x1xi32> -> tensor<2x2x2x2x2xi32>
    %139 = arith.xori %135, %138 : tensor<2x2x2x2x2xi32>
    %140 = tt.bitcast %139 : tensor<2x2x2x2x2xi32> -> tensor<2x2x2x2x2xf32>
    %141 = arith.cmpf ogt, %134, %140 : tensor<2x2x2x2x2xf32>
    %142 = arith.cmpi ne, %21, %cst_5 : tensor<1x1x1x1x2xi32>
    %143 = tt.broadcast %142 : tensor<1x1x1x1x2xi1> -> tensor<2x2x2x2x2xi1>
    %144 = arith.cmpi ne, %141, %143 : tensor<2x2x2x2x2xi1>
    %145 = arith.select %144, %140, %134 : tensor<2x2x2x2x2xi1>, tensor<2x2x2x2x2xf32>
    %146 = tt.reshape %145 : tensor<2x2x2x2x2xf32> -> tensor<2x16xf32>
    %147 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %148 = tt.addptr %147, %5 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %149 = tt.broadcast %148 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x16x!tt.ptr<f32>>
    %150 = tt.addptr %149, %9 : tensor<2x16x!tt.ptr<f32>>, tensor<2x16xi32>
    tt.store %150, %146, %11 : tensor<2x16x!tt.ptr<f32>>
    tt.return
  }
}

