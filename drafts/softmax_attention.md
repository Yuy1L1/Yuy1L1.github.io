# Learning CUDA Softmax Attention

This is a living learning note.

I am writing down what I understand as I go: first correctness, then
performance. Each level is a slightly different answer to the same question:
why is correct kernel composition not enough for attention?

This is made possible with the hardware support from the [UCCL project](https://github.com/uccl-project). We are going to start with challenges based on [LeetGPU](https://leetgpu.com/challenges).

## Table of contents

1. What is softmax attention?
2. Shape bookkeeping
3. First decomposition: compose kernels I already know
4. Why this feels familiar
5. Level 0: explicit staged baseline
6. Where the baseline gets stuck
7. Level 1: fusing row-wise softmax
8. What the fused softmax fixed
9. What is still materialized
10. Interlude: kernel launches as synchronization boundaries
11. Interlude: row-wise softmax is a reduction problem
12. Horizontal vs vertical decomposition
13. Level 2: Online softmax, the missing trick
14. What fused attention changes
15. Level 3: toward a FlashAttention-style kernel
16. Bugs I hit
17. Reading notes: TBD
18. Experiments and benchmarks
19. Open questions

## What is softmax attention?

Softmax attention computes:

```
output = softmax(Q K^T / sqrt(d)) V
```

where the softmax is applied row by row.

For the LeetGPU version of the problem, the implementation uses only GPU native
features, keeps the given `solve` signature, and writes the final result into
`output`.

## Shape bookkeeping

Before writing kernels, I need the shapes to be boringly clear:

```
Q:      M x d
K:      N x d
K^T:    d x N
QK^T:   M x N
V:      N x d
output: M x d
```

This means the intermediate attention score matrix has shape `M x N`. That
fact is going to matter a lot.

## First decomposition: compose kernels I already know

At first, softmax attention looks like a build-up from previous kernels:

1. Transpose `K`.
2. Multiply `Q` and `K^T`.
3. Scale by `1 / sqrt(d)`.
4. Apply softmax row by row.
5. Multiply the softmax result by `V`.

The softmax step itself decomposes into familiar pieces:

1. For each row, find the max.
2. Subtract the max and apply `expf`.
3. Sum the row.
4. Divide each entry by the row sum.

So the first decomposition is useful for correctness. It lets me reuse ideas
from transpose, reduction, softmax, and GEMM.

## Why this feels familiar

The transpose part is basically the transpose blog again. Intuitively, we can
write something like:

```cpp
__global__ void transpose(const float* __restrict__ input, float* __restrict__ transposed, int M, int N) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    // suppose I am given a matrix of M * N, the transposed result will be N * M
    if (row < M && col < N) {
        transposed[col * M + row] = input[row * N + col];
    }
}
```

This version exposes two familiar issues.

First, it does not use shared memory as a staging tile. Second, the write:

```
transposed[col * M + row]
```

is strided through global memory. The reads are coalesced, but neighboring
threads write to addresses that are `M` elements apart, so the writes are not
coalesced.

The classic fix is to load a tile into shared memory with coalesced reads, then
read that tile back transposed so the global writes are also coalesced. This is
the same mental move as the transpose blog: spend shared memory to improve the
global memory access pattern.

## Level 0: explicit staged baseline

The following code is my baseline attempt. I am intentionally keeping it as a
composition of explicit stages:

```
transpose K
GEMM: Q * K^T
scale
row-wise softmax
GEMM: softmax_result * V
```


```cpp
#include <cuda_runtime.h>
#include <math.h>
#define FULL_MASK 0xffffffff 

// first step: transpose K
// second step: multiply Q and K_t
// third step: elementwise division
// fourth step: softmax
// fifth step: GEMM

#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define FULL_MASK 0xffffffff 

// first step: transpose K
// second step: multiply Q and K_t
// third step: elementwise division
// fourth step: softmax
// fifth step: GEMM

__global__ void transpose(const float* __restrict__ input,
                          float* __restrict__ transposed,
                          int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // K row, 0..N-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K col, 0..d-1

    if (row < N && col < d) {
        transposed[col * N + row] = input[row * d + col];
    }
}


__global__ void matrix_multiplication(
    const float* __restrict__ input_A, 
    const float* __restrict__ input_B, 
    float* __restrict__ gemm_result, 
    int dim_A, 
    int dim_B, 
    int dim_C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    if (row < dim_A && col < dim_B) {
        for(int k = 0; k < dim_C; k++){
            sum += input_A[row * dim_C + k] * input_B[k * dim_B + col];
        }
        gemm_result[row * dim_B + col] = sum;
    }
    
}

__global__ void division(float* __restrict__ input, int M, int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    if (row  < M && col < N) {
        input[row * N + col] =  input[row * N + col] / sqrtf(d);
    }
}

__global__ void find_max_per_block(float* __restrict__ input, int input_size, float* __restrict__ blockwise) {
    // apply the warp trick here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;

    int lane_id = tid % 32; 
    int warp_id = tid / 32; // which warp group I am in

    float val = 0.0f;

    if (i < input_size) {
        val = input[i];
    } else {
        val = -FLT_MAX;
    }
    __syncthreads();

    int stride = 32 / 2 ;

    while(stride > 0) {
        val = fmax(val, __shfl_down_sync(FULL_MASK, val, stride));
        __syncthreads();
        stride /= 2;
    }

    __shared__ float shared[8]; // currently hard coded
    
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < 8) ? shared[lane_id] : -FLT_MAX;
        val = fmax(val, __shfl_down_sync(FULL_MASK, val, 4));
        val = fmax(val, __shfl_down_sync(FULL_MASK, val, 2));
        val = fmax(val, __shfl_down_sync(FULL_MASK, val, 1));
        if (lane_id == 0) {
            // found the max, return
            blockwise[blockIdx.x] = val;
        }
    }
}

// softmax decomposition

__global__ void find_max(float* __restrict__ blockwise, int input_size, float* max_result) {
    // reduction across the block
    int blockwise_size = (input_size + blockDim.x -1) / blockDim.x;

    int i = threadIdx.x;
    __shared__ float shared[256];

    if (i < blockwise_size) {
        shared[i] = blockwise[i];
    } else {
        shared[i] = -FLT_MAX;
    }
    __syncthreads();

    int stride = 256 / 2;
    while(stride > 0) {
        if (i < stride) {
            shared[i] = fmax(shared[i], shared[i + stride]);
        }
        __syncthreads();
        stride /= 2;
    }
    if (i == 0) { max_result[0] = shared[0]; }
}

__global__ void subtract(float* __restrict__ input, int input_size, float* max_result) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i  < input_size) {
        input[i] = expf(input[i] - max_result[0]);
    }
}


__global__ void reduction_per_block(float* __restrict__ input, int input_size, float* __restrict__ blockwise) {
    // apply the warp trick here
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid =  threadIdx.x;

    int lane_id = tid % 32; 
    int warp_id = tid / 32; // which warp group I am in

    float val = 0.0f;

    if (i < input_size) {
        val = input[i];
    }
    __syncthreads();

    int stride = 32 / 2 ;

    while(stride > 0) {
        val += __shfl_down_sync(FULL_MASK, val, stride);
        __syncthreads();
        stride /= 2;
    }

    __shared__ float shared[8]; // currently hard coded
    
    // now that per warp sum is at lane_id = 0
    if (lane_id == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < 8) ? shared[lane_id] : 0.0f;
        val += __shfl_down_sync(FULL_MASK, val, 4);
        val += __shfl_down_sync(FULL_MASK, val, 2);
        val += __shfl_down_sync(FULL_MASK, val, 1);
        if (lane_id == 0) {
            // we have aggregated the reduction into one val. 
            // return
            blockwise[blockIdx.x] = val;
        }
    }
}

__global__ void reduction(float* __restrict__ blockwise, int input_size, float* __restrict__ reduction_result) {
    // reduction across the block
    int blockwise_size = (input_size + blockDim.x -1) / blockDim.x;

    int i = threadIdx.x;
    __shared__ float shared[256];

    if (i < blockwise_size) {
        shared[i] = blockwise[i];
    } else {
        shared[i] = 0.0f;
    }
    __syncthreads();

    int stride = 256 / 2;
    while(stride > 0) {
        if (i < stride) {
            shared[i] += shared[i + stride];
        }
        __syncthreads();
        stride /= 2;
    }
    if (i == 0) { reduction_result[0] = shared[0]; }
}

__global__ void divide(float* __restrict__ input, int N, float* reduction_result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        input[i] = input[i] / reduction_result[0];
    }
}


// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // now, 1D launch
    int TPB = 256;
    int BPG = (N + TPB - 1) / TPB;

    float *transposed, *gemm_result, *blockwise, *max_result, *reduction_result;
    cudaMalloc(&transposed, sizeof(float) * N * d);
    cudaMalloc(&gemm_result, sizeof(float) * M * N);
    cudaMalloc(&blockwise, sizeof(float) * BPG);
    cudaMalloc(&max_result, sizeof(float));
    cudaMalloc(&reduction_result, sizeof(float));

    dim3 transposeBlock(16, 16);
    dim3 transposeGrid((d + 15) / 16, (N + 15) / 16);

    transpose<<<transposeGrid, transposeBlock>>>(K, transposed, N, d);

    cudaDeviceSynchronize();

    matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(Q, transposed, gemm_result, M, N, d);

    cudaDeviceSynchronize();

    division<<<blocksPerGrid, threadsPerBlock>>>(gemm_result, M, N, d);

    cudaDeviceSynchronize();

    // this is the problem. i am doing softmax per row. so now given gemm result is M * N.
    // I need to do this M times
    for(int i = 0; i < M; i++) {
        float* row_ptr = gemm_result + i * N;
        find_max_per_block<<<BPG, TPB>>>(row_ptr, N, blockwise);
            cudaDeviceSynchronize();

        find_max<<<1, TPB>>>(blockwise, N, max_result);
        cudaDeviceSynchronize();

        subtract<<<BPG, TPB>>>(row_ptr, N, max_result);
        cudaDeviceSynchronize();

        reduction_per_block<<<BPG, TPB>>>(row_ptr , N, blockwise);
        cudaDeviceSynchronize();

        reduction<<<1, TPB>>>(blockwise, N, reduction_result);
        cudaDeviceSynchronize();

        divide<<<BPG, TPB>>>(row_ptr, N, reduction_result);
    }

    dim3 finalGrid((d + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication<<<finalGrid, threadsPerBlock>>>(gemm_result, V, output, M, d, N);
    cudaDeviceSynchronize();

    cudaFree(transposed);
    cudaFree(gemm_result);
    cudaFree(blockwise);
    cudaFree(max_result);
    cudaFree(reduction_result);

}

```

The above kernel works just fine. And it is my first softmax kernel. One of its most obivous problem is the host-side loop.

On a Nvidia Tesla T4 GPU, for M = 512 and n = 256, this kernel takes 24.91 ms to finish.
Unless stated otherwise, the result is going to be measured with this M and N with each entry FP32.
We are only at 6.1th percentile.

<div style="border: 1px solid #d0d7de; border-left: 4px solid #0969da; padding: 0.85rem 1rem; margin: 1rem 0; border-radius: 6px; background: #f6f8fa;">
  <strong>Crux 1.</strong> Can we avoid host-side synchronization and extra global-memory trips by letting one kernel finish each row's softmax?
</div>

The answer is yes.

## Where the baseline gets stuck

The baseline bottleneck is the host-side loop over `M` rows. After `QK^T`, the
score matrix is `M x N`, but softmax is not one global operation. It is `M`
independent row-wise softmaxes, and each row needs:

1. a max reduction,
2. an elementwise `expf`,
3. a sum reduction,
4. an elementwise division.

Level 0 does all of that correctly, but each row becomes a chain of kernel
launches, temporary buffers, global-memory trips, and synchronization points.
The CPU keeps issuing the next phase while the GPU waits at boundaries. This is
where attention stops looking like a clean composition of known kernels and
starts asking for fusion.

## Level 1: fusing row-wise softmax

The next step was to remove the host-side loop over rows and fuse the row-wise
softmax into one kernel. Each block owns one row of the `M x N` score matrix.
Inside the block, the max, exponentiation, sum, and division happen without
round-tripping through separate global-memory buffers.

This is still not full fused attention. It only fuses the softmax part.

The change is easiest to see as a diff. The old path decomposed softmax into
multiple helper kernels and then repeated that decomposition for every row.
The new path replaces that row loop with a single `fused_softmax` launch:

```diff
- // softmax decomposition
- __global__ void find_max(...)
- __global__ void subtract(...)
- __global__ void reduction_per_block(...)
- __global__ void reduction(...)
- __global__ void divide(...)
-
+ __global__ void fused_softmax(float* __restrict__ input, int N) {
+     // one block owns one row
+     // phase 1: row max reduction
+     // phase 2: exp(row - max)
+     // phase 3: row sum reduction
+     // phase 4: normalize and write back
+ }

  extern "C" void solve(...) {
      ...
-     for(int i = 0; i < M; i++) {
-         float* row_ptr = gemm_result + i * N;
-         find_max_per_block<<<BPG, TPB>>>(row_ptr, N, blockwise);
-         find_max<<<1, TPB>>>(blockwise, N, max_result);
-         subtract<<<BPG, TPB>>>(row_ptr, N, max_result);
-         reduction_per_block<<<BPG, TPB>>>(row_ptr, N, blockwise);
-         reduction<<<1, TPB>>>(blockwise, N, reduction_result);
-         divide<<<BPG, TPB>>>(row_ptr, N, reduction_result);
-     }
+     fused_softmax<<<M, 1024, N * sizeof(float)>>>(gemm_result, N);
      ...
  }
```

Here is the exact code, folded so the main narrative stays easy to scan.

<details markdown="1">
<summary>Full Level 1 code</summary>

```cpp
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define FULL_MASK 0xffffffff 

// first step: transpose K
// second step: multiply Q and K_t
// third step: elementwise division
// fourth step: softmax
// fifth step: GEMM

__global__ void transpose(const float* __restrict__ input,
                          float* __restrict__ transposed,
                          int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // K row, 0..N-1
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K col, 0..d-1

    if (row < N && col < d) {
        transposed[col * N + row] = input[row * d + col];
    }
}


__global__ void matrix_multiplication(
    const float* __restrict__ input_A, 
    const float* __restrict__ input_B, 
    float* __restrict__ gemm_result, 
    int dim_A, 
    int dim_B, 
    int dim_C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0f;

    if (row < dim_A && col < dim_B) {
        for(int k = 0; k < dim_C; k++){
            sum += input_A[row * dim_C + k] * input_B[k * dim_B + col];
        }
        gemm_result[row * dim_B + col] = sum;
    }
    
}

__global__ void division(float* __restrict__ input, int M, int N, int d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    if (row  < M && col < N) {
        input[row * N + col] =  input[row * N + col] / sqrtf(d);
    }
}

__global__ void fused_softmax(float* __restrict__ input, int N) {
    // this is my first fused kernel
    // Intermediate results live in registers and shared memory.
    // The rule is: if data is produced and consumed in the same logical step, 
    // it should never touch global memory.

    extern __shared__ float row[];
    __shared__ float max_res;
    __shared__ float warp_scratch[32];
    __shared__ float reduction_res;
    int tid = threadIdx.x;

    int lane_id = tid % 32; 
    int warp_id = tid / 32;

    int stride = 32 / 2;

    for (int i = tid; i < N; i += blockDim.x) {
        row[i] = input[blockIdx.x * N + i];
    }

    __syncthreads();

    // phase 1: each thread finds local max over its strided elements
    float val = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x)
        val = fmax(val, row[i]);

    // NOW reduce val across all threads in the block -> one max_res
    // ... warp shuffle + shared mem reduction ...
    while(stride > 0) {
        val = fmax(val, __shfl_down_sync(FULL_MASK, val, stride));
        stride /= 2;
    }
    // now lane_id == 0 of each warp has the max

    if(lane_id == 0) {
        warp_scratch[warp_id] = val;
    }
    __syncthreads();
    // warp reduce -> shared[warp_id] = warp max
    // warp 0 reduces shared[] -> max_res
    if(warp_id == 0) {
        val = warp_scratch[lane_id];
        stride = 16;
        while(stride > 0) {
            val = fmax(val, __shfl_down_sync(FULL_MASK, val, stride));
            stride /= 2;
        }
        // write back
        if(lane_id == 0) {
            max_res = val;
        }
    }
    __syncthreads();

    // phase 2: now everyone can read max_res
    for (int i = tid; i < N; i += blockDim.x)
        row[i] = expf(row[i] - max_res);
    __syncthreads();

    // phase 3: each thread finds local sum over its strided elements
    val = 0.0f;
    for (int i = tid; i < N; i += blockDim.x)
        val += row[i];

    // NOW reduce val across all threads in the block -> one max_res
    // ... warp shuffle + shared mem reduction ...
    lane_id = tid % 32; 
    warp_id = tid / 32;

    stride = 32 / 2;

    while(stride > 0) {
        val += __shfl_down_sync(FULL_MASK, val, stride);
        stride /= 2;
    }
    // now lane_id == 0 of each warp has the reduction of each warp group

    if(lane_id == 0) {
        warp_scratch[warp_id] = val;
    }
    __syncthreads();
    // warp reduce -> shared[warp_id] = warp reduction
    // warp 0 reduces shared[] -> max_res
    if(warp_id == 0) {
        val = warp_scratch[lane_id];
        stride = 16;
        while(stride > 0) {
            val += __shfl_down_sync(FULL_MASK, val, stride);
            stride /= 2;
        }
        // write back
        if(lane_id == 0) {
            reduction_res = val;
        }
    }
    __syncthreads();

    // phase 4
    for (int i = tid; i < N; i += blockDim.x)
        input[blockIdx.x * N + i] = row[i] / reduction_res;

}

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float *transposed, *gemm_result;
    cudaMalloc(&transposed, sizeof(float) * N * d);
    cudaMalloc(&gemm_result, sizeof(float) * M * N);

    dim3 transposeBlock(16, 16);
    dim3 transposeGrid((d + 15) / 16, (N + 15) / 16);

    transpose<<<transposeGrid, transposeBlock>>>(K, transposed, N, d);

    cudaDeviceSynchronize();

    matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(Q, transposed, gemm_result, M, N, d);

    cudaDeviceSynchronize();

    division<<<blocksPerGrid, threadsPerBlock>>>(gemm_result, M, N, d);

    cudaDeviceSynchronize();

    fused_softmax<<<M, 1024, N * sizeof(float)>>>(gemm_result, N);


    dim3 finalGrid((d + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrix_multiplication<<<finalGrid, threadsPerBlock>>>(gemm_result, V, output, M, d, N);
    cudaDeviceSynchronize();

    cudaFree(transposed);
    cudaFree(gemm_result);

}
```

</details>

Almost immediately, we see a great performance boost on the exact same GPU with this fused softmax
kernel.

We are now at 75.4th percentile, spending only 0.49ms on the same metric.

## What the fused softmax fixed

The fused softmax kernel removes the host-side loop over `M` rows. Instead of
launching several kernels per row, it launches one block per row:

```
fused_softmax<<<M, 1024, N * sizeof(float)>>>(gemm_result, N);
```

Each block owns one full row of the score matrix. That block can find the row
max, exponentiate, reduce the row sum, and normalize the row without asking the
CPU to launch the next phase.

This fixed one obvious problem: the row-wise softmax no longer forces an
outer-loop on the host.

TODO: measure the difference in kernel count and host-side synchronization cost
between the explicit baseline and the fused-softmax version.

## What is still materialized

The first decomposition writes the full score matrix:

```
scores = Q K^T
```

with shape:

```
scores: M x N
```

Then it scales `scores`, softmaxes `scores`, and eventually multiplies the
softmax result by `V`.

This is correct, but it materializes an intermediate matrix that the final
algorithm does not actually need to keep around. The final output only needs:

```
softmax(scores) V
```

In the code above, this is `gemm_result`:

```
cudaMalloc(&gemm_result, sizeof(float) * M * N);
```

To materialize something means to write the whole intermediate object to global
memory so it exists there in full. For `M = N = 4096` and FP32, the attention
score matrix alone costs:

```
4096 * 4096 * 4 bytes = 64 MB
```

And the cost is not just allocation. The staged pipeline writes `gemm_result`,
reads it for scaling and softmax, writes it again, and reads it again for the
final multiplication by `V`.

<div style="border: 1px solid #d0d7de; border-left: 4px solid #0969da; padding: 0.85rem 1rem; margin: 1rem 0; border-radius: 6px; background: #f6f8fa;">
  <strong>Crux 2.</strong> Can I compute the final output without writing the whole <code>M x N</code> attention matrix to global memory?
</div>

The answer is again, yes.

## Interlude: kernel launches as synchronization boundaries

The baseline also launches many kernels:

```
transpose
GEMM
scale
find row max
subtract / exp
row sum
divide
GEMM
```

Each launch boundary can be useful for reasoning, because the next kernel sees
the completed output of the previous kernel. But these boundaries also mean
more global memory traffic and more scheduling overhead.

This connects back to the reduction blog: a kernel launch can act like a global
ordering point, but relying on many launches is not free.

TBD: count the exact number of reads/writes to the `M x N` score matrix in the
baseline and fused-softmax versions.

## Interlude: row-wise softmax is a reduction problem

Row-wise softmax is where the reduction blog comes back:

```
softmax(row_i) = exp(row_i - max(row_i)) / sum(exp(row_i - max(row_i)))
```

For each row, I need a max reduction and a sum reduction. That means attention
is not just GEMM plus pointwise math. It is GEMM plus per-row reductions plus a
second GEMM.

The fused softmax kernel works because one block sees an entire row at once. But
if I tile `QK^T`, a block only sees part of a row at a time. Then the row max
and row sum are not immediately available. That is the online-softmax problem.

## Horizontal vs vertical decomposition

At this point, I started to see two ways to decompose attention.

Horizontal decomposition is what I had been doing:

```
op1: transpose K        -> all of K^T
op2: matmul QK^T        -> all of M x N
op3: softmax            -> all of M x N
op4: matmul with V      -> all of M x d
```

Each operation runs to completion across the whole input before the next one
starts. This is simple to reason about, but each boundary is a round trip
through global memory.

Vertical decomposition is what tiling is trying to do:

```
tile 1: QK^T -> softmax -> multiply by V -> output contribution
tile 2: QK^T -> softmax -> multiply by V -> output contribution
tile 3: QK^T -> softmax -> multiply by V -> output contribution
```

Each tile moves through multiple operations while it is still in fast memory.
The hard part is that operations are not always independent across tiles.
Softmax is the annoying one because it seems to need the global max and global
sum of the whole row.

## Level 2: Online softmax, the missing trick

<div style="border: 1px solid #d0d7de; border-left: 4px solid #0969da; padding: 0.85rem 1rem; margin: 1rem 0; border-radius: 6px; background: #f6f8fa;">
  <strong>Crux 3.</strong> If I process only one tile of a row, I do not know the final row max yet. If a later tile has a larger max, do I need to recompute the earlier exponentials?
</div>

The answer is no. I need to rescale.

Suppose after tile 1:

```
m1 = max(tile1)
s1 = sum(exp(x - m1)) for x in tile1
```

Then tile 2 arrives and has a larger max `m2`. The correct old contribution
under the new max is:

```
s1_corrected = s1 * exp(m1 - m2)
```

because:

```
exp(x - m1) * exp(m1 - m2) = exp(x - m2)
```

So the running update can be written as:

```
m_new = max(m_old, tile_max)
s_new = s_old * exp(m_old - m_new) + tile_sum * exp(tile_max - m_new)
```

The same rescaling applies to the running output accumulator. That is the step
that makes it possible to fuse `QK^T`, softmax, and multiplication by `V`
without materializing the whole `M x N` score matrix.

## What fused attention changes

The fused-attention idea is not just "make the code shorter." The important
change is that we avoid writing the full attention matrix to global memory.

Instead of:

1. compute all scores
2. store all scores
3. softmax all scores
4. store all probabilities
5. multiply probabilities by V


the fused version tries to do this in tiles:

1. load a block of Q
2. for blocks of K and V:

    compute partial QK^T scores

    update online softmax state

    accumulate partial output

3. write final output

This is where FlashAttention enters the story.

## Level 3: toward a FlashAttention-style kernel

The next code block is my first attempt at moving from "fuse only softmax" to
"fuse the attention pipeline." It uses tiles of `K` and `V`, computes a tile of
`QK^T`, updates online softmax state, and accumulates the output.

This is still a learning implementation, not FlashAttention-2 or a production
kernel.

<div style="border: 1px solid #d0d7de; border-left: 4px solid #0969da; padding: 0.85rem 1rem; margin: 1rem 0; border-radius: 6px; background: #f6f8fa;">
  <strong>Crux 4.</strong> The full <code>M x N</code> attention matrix no longer has to exist in global memory.
</div>

Here is the code

```cpp
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#define FULL_MASK 0xffffffff 
#define TILE_SIZE 32
#define MAX_D 128

__global__ void flash_attention(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int M, int N, int d
    ) 
    {
        // using tiling
        int tid = threadIdx.x;
        int row = blockIdx.x;
        int lane_id = tid % 32; 
        int warp_id = tid / 32;

        // these are the initialization.
        // but in actual code writing, I leave this part blank and
        // fill in as my implementation goes.

        __shared__ float K_tile[TILE_SIZE * MAX_D];
        __shared__ float V_tile[TILE_SIZE * MAX_D];
        __shared__ float S[TILE_SIZE];  // dot products for this tile
        __shared__ float warp_scratch[32];
        __shared__ float tile_max;
        __shared__ float tile_sum;

        int stride = 32 / 2;

        // running state (per thread, in registers)
        float m_old = -FLT_MAX;
        float s_old = 0.0f;
        // output accumulator: each thread owns a slice of the d dimension
        float o[MAX_D] = {0.0f};  // we'll deal with this sizing in a moment

        int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

        for (int j = 0; j < num_tiles; j++) {
            for(int i = tid; i < TILE_SIZE * d; i+= blockDim.x) {
                // 1. load K_tile and V_tile from global mem
                int local_row = i / d;
                int col = i % d;
                int global_row = j * TILE_SIZE + local_row;
                if(global_row < N) {
                    K_tile[i] = K[global_row * d + col];
                    V_tile[i] = V[global_row * d + col]; 
                }
                else {
                    K_tile[i] = 0.0f;
                    V_tile[i] = 0.0f; 
                }
            }
            __syncthreads();
            // 2. compute S[t] = dot(Q[row], K_tile[t]) for t in 0..TILE_SIZE
            // 3. scale S[t] /= sqrt(d)
            for (int t = tid; t < TILE_SIZE; t += blockDim.x) {
                int global_row = j * TILE_SIZE + t;
                if (global_row < N) {
                    float dot = 0.0f;
                    for (int k = 0; k < d; k++)
                        dot += Q[row * d + k] * K_tile[t * d + k];
                    S[t] = dot / sqrtf((float)d);
                } else {
                    S[t] = -FLT_MAX;  // padding
                }
            }
            __syncthreads();

            // 4. online softmax update: find tile max, rescale m/s/o
            // phase 1: each thread finds local max over its strided elements
            float val = -FLT_MAX;
            for (int t = tid; t < TILE_SIZE; t += blockDim.x) {
                val = fmax(val, S[t]);
            }
                
            
            // NOW reduce val across all threads in the block -> one max_res
            // ... warp shuffle + shared mem reduction ...
            for (int stride = 16; stride > 0; stride /= 2)
                val = fmax(val, __shfl_down_sync(FULL_MASK, val, stride));
            // now lane_id == 0 of each warp has the max
            if(lane_id == 0) {
                warp_scratch[warp_id] = val;
            }
            __syncthreads();
            // warp reduce -> shared[warp_id] = warp max
            // warp 0 reduces shared[] -> max_res
            if(warp_id == 0) {
                val = (lane_id < (blockDim.x / 32)) ? warp_scratch[lane_id] : -FLT_MAX;
                stride = 16;
                while(stride > 0) {
                    val = fmax(val, __shfl_down_sync(FULL_MASK, val, stride));
                    stride /= 2;
                }
                // write back
                if(lane_id == 0) {
                    tile_max = val;
                }
            }
            __syncthreads();

            float m_new = fmax(m_old, tile_max);

            val = 0.0f;
            for(int t = tid; t < TILE_SIZE; t+=blockDim.x) {
                S[t] = expf(S[t] - m_new);
                val += S[t];
            }

            // reduce
            for (int stride = 16; stride > 0; stride /= 2)
                val += __shfl_down_sync(FULL_MASK, val, stride);
            // now lane_id == 0 of each warp has the reduction of each warp group

            if(lane_id == 0) {
                warp_scratch[warp_id] = val;
            }
            __syncthreads();
            // warp reduce -> shared[warp_id] = warp reduction
            // warp 0 reduces shared[] -> max_res
            if(warp_id == 0) {
                val = (lane_id < (blockDim.x / 32)) ? warp_scratch[lane_id] : 0.0f;
                stride = 16;
                while(stride > 0) {
                    val += __shfl_down_sync(FULL_MASK, val, stride);
                    stride /= 2;
                }
                // write back
                if(lane_id == 0) {
                    tile_sum = val;
                }
            }
            __syncthreads();

            float scale_old = expf(m_old - m_new);
            float s_new = s_old * scale_old + tile_sum;

            // rescale
            for (int i = tid; i < d; i += blockDim.x) {
                o[i] *= scale_old;
                for (int t = 0; t < TILE_SIZE; t++) {
                    o[i] += S[t] * V_tile[t * d + i];
                }
            }

            m_old = m_new;
            s_old = s_new;
            __syncthreads();

        }

        for(int i = tid; i<d; i+= blockDim.x) {
            output[row * d + i] = o[i] / s_old;
        }

    }

// Q, K, V, output are device pointers
extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int M, int N, int d) {
    int threads = 256;

    size_t shared_mem = 0;
    flash_attention<<<M, threads>>>(Q, K, V, output, M, N, d);
    cudaDeviceSynchronize();

}

```

## Interlude: Bugs I hit trying to go from partially fused to fully fused Flash-Attention Style kernel

These are the bugs I hit before reaching a compilable and runnable fused
attention-style kernel.

### Indexing

- Used `input` instead of `K` and `V`.
- Used `blockIdx.x * blockDim.x + threadIdx.x` inside a kernel where
  `blockIdx.x` means "which row", mixing two coordinate systems.
- Used `i` from an outer loop inside an inner loop where `t` was needed.
- Strided the inner `V` accumulation loop by `blockDim.x`, even though every
  thread needs to see all of `S[]`.

### Shared memory

- Tried to declare `extern __shared__ float tile[][]`, but 2D extern shared
  arrays do not exist in CUDA.
- Declared `__shared__` variables mid-function after executable code.
- Declared `warp_scratch[32]`, filled only 8 slots for 256 threads, then read
  all 32 in the warp-0 reduction. The uninitialized lanes corrupted results.
- Declared the same shared array twice for two reduction phases.

### Reductions

- Put `__syncthreads()` inside warp shuffle loops. This is unnecessary for the
  warp shuffle itself and can make reasoning harder.
- Reused `stride` after it had already been reduced to 0.
- Used `0.0f` as the neutral element for padded score entries. The correct value
  is `-FLT_MAX`, otherwise padding contributes fake attention weight.

### Logic

- Replaced `V` accumulation with score recomputation, so `o[i]` was never
  updated.
- Confused `s_old` and `s_new` while naming the running softmax state.
- Reused temporary variables across phases without resetting them.

### Signature / launch

- Kept dead scratch parameters after fusing kernels.
- Missed arguments in kernel launches.
- Used `float o[MAX_D]`, which requires `MAX_D` to be a compile-time constant.

The two hardest bugs to spot were:

1. The uninitialized `warp_scratch` lanes. The output looked mostly right, but
   had a small max error hiding in the middle of large arrays.
2. The padding bug. Zeroing `K_tile` and `V_tile` for out-of-bounds rows felt
   correct, but the real fix was setting `S[t] = -FLT_MAX` at the score stage.



## Experiments and benchmarks (WIP)

Some quick benchmarking numbers:

On Nvidia Tesla T4:

The naive fused-attention-style kernel takes roughly 3.75 ms for `M = 512` and
`N = 256` in FP32, while the softmax-fused-only version takes roughly 0.49 ms.

On Nvidia H100: The fused-attention-style kernel takes roughly 0.18 ms, while the
softmax-fused-only version takes roughly 0.13 ms.

The fused-attention-style kernel is doing more work than the softmax-only
kernel, so this comparison is not apples-to-apples yet. The better comparison
is against the fully staged baseline and PyTorch's
`scaled_dot_product_attention`.

TBD: compare the baseline staged implementation with PyTorch
`scaled_dot_product_attention`.

TBD: measure memory footprint for different `M`, `N`, and `d`.

TBD: do an ablation study. Which part of the fused kernel contributes most to
the speedup, and which part is now the bottleneck?

## Open questions

What tile sizes make sense for B300 and GH200?

Note that Blackwell and Hopper has enormous L2 Cache, comparing to the Ampere architecture.

Principles: dentify the actual bottleneck before choosing tile sizes. Is the current
implementation limited by HBM traffic, L2 behavior, shared memory, occupancy, or
register pressure?

TBD: try BF16 after the FP32 version is understood. Quantization changes both
performance and numerical behavior, so it should be a separate experiment.

How much of this should I implement myself before switching to reading
FlashAttention/Triton/CUTLASS source?

I tried to read the FlashAttention paper too early and bounced off it. Going
back to the naive decomposition helped: first I fused only softmax, then I tried
to fuse the rest of the pipeline. That made the paper's core idea feel much more
concrete.

## Further Readings and References

Read the Triton fused-attention tutorial because it exposes variables like
`m_i`, `l_i`, and `acc`, which map directly to the running softmax state.

Read CUTLASS/CuTe attention examples for the hardware-aware tiled version.
