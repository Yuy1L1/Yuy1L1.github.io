# Learning CUDA Transpose

This is a living learning note. 

I am writing down what I understand as I go:
first correctness, then performance. Each level is a slightly different answer
to the same question: how do we move data from one layout to another without
making global memory access terrible?

This is made possible with the hardware support from the [UCCL project](https://github.com/uccl-project). We are going to start with challenges based on [LeetGPU](https://leetgpu.com/challenges).


First, what is transpose? Intuitively, this is what we write in CPU:

```
let input be M * N and output be N * M

for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
        output[j][i] = input[i][j];
    }
}
```

Since we are in CUDA, we apply the same principle from reduction: think in terms
of threads. What is the job of each thread? For the first version, one thread
will read one input element and write it to the transposed output position.

### Level 0: Naive version

### Interlude 0: 1D launch and 2D launch

Before, when we want to launch kernels, on the host side, we have something like this:
```
int threadsPerBlock = 256;
int blocksPerGrid  = (N + threadsPerBlock - 1) / threadsPerBlock;

my_kernel<<<blocksPerGrid, threadsPerBlock>>>(my params);
```

But for this problem, we are going to launch the kernel in 2D fashion. A 2D
launch does not create a different kind of thread. It gives the same CUDA grid a
2D coordinate system. That coordinate system matches the matrix: `threadIdx.x`
naturally maps to columns, and `threadIdx.y` naturally maps to rows.

So even if we are launching the same set of 256 threads, if we arrange it in 2D, then we get 

```
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
```

I could flatten the matrix into one global index and recover `row` and `col`
with division and modulo. But for transpose, row and column are both first-class
concepts, so a 2D launch makes the code easier to read and debug.


Eventually, we will get something functional but not performant.
```
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}
```

### Interlude 1: coalesced reads, strided writes

The naive kernel is correct, but its memory access pattern is lopsided.

For neighboring threads in the same row, `col` changes by 1. So the input read:

```
input[row * cols + col]
```

is contiguous. Adjacent threads read adjacent addresses. This is good; the GPU
can combine those loads into efficient memory transactions.

But the output write:

```
output[col * rows + row]
```

is different. If `row` is fixed and `col` changes by 1, then the output address
jumps by `rows` elements each time. Adjacent threads are no longer writing
adjacent addresses. They are writing a strided pattern through memory.

This is the first real lesson of transpose for me: correctness is just index
math, but performance is about whether neighboring threads touch neighboring
addresses.

### Level 1: shared-memory tiled transpose

The next step is to use shared memory as a staging area. Shared memory does not
make transpose fast by magic. It lets me separate two concerns:

```
how I read from global memory
how I write to global memory
```

The goal is to load a tile in a coalesced way, synchronize the block, and then
write the tile out in its transposed position.


Let us start with an easy 4×4 example:
```
Input (4×4):          Output (4×4 transposed):
[1  2  3  4            [1  5  9  13
5  6  7  8             2  6  10 14
9  10 11 12            3  7  11 15
13 14 15 16]           4  8  12 16]
```
And our blocks are only 2*2 big. So the 4 tiles and what happens to each:
```
Input tile positions:        Output tile positions:
[block(0,0)] [block(1,0)]    [block(0,0)] [block(0,1)]
 1  2          3  4           1  5          9  13
 5  6          7  8           2  6          10 14


[block(0,1)] [block(1,1)]    [block(1,0)] [block(1,1)]
 9  10         11 12          3  7          11 15
 13 14         15 16          4  8          12 16
```
By observing this example, we can realize that two things happen simultaneously:

Within the tile: numbers get transposed (shared[x][y] instead of shared[y][x])
The tile itself moves: block(0,1) in input becomes block(1,0) in output, which is exactly why we swap blockIdx.x and blockIdx.y when computing out_row and out_col
```
// WHERE the tile lands in output = swap block indices
int out_col = blockIdx.y * TILE + threadIdx.x;  // blockIdx.y not .x
int out_row = blockIdx.x * TILE + threadIdx.y;  // blockIdx.x not .y

// WHAT gets written = swap thread indices into shared
output[out_row * rows + out_col] = shared[threadIdx.x][threadIdx.y];
```
Both swaps are necessary — get only one and the output is wrong in a different way each time.

```
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* input, float* output, int rows, int cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float shared[16][16];

    // load from global to SMEM
    if(row < rows && col < cols) {
        // then how to make sure it is inside the 16 * 16 grid? no idea
        shared[threadIdx.y][threadIdx.x] = input[row * cols + col];
    }
    else {
        shared[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    int out_col = blockDim.x * blockIdx.y + threadIdx.x;
    int out_row = blockDim.y * blockIdx.x + threadIdx.y;
    
    if(out_col < rows && out_row < cols) {
        // now, transpose
        output[out_row * rows + out_col] = shared[threadIdx.x][threadIdx.y];
    }

}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

```

Note to add later: a code diff from Level 0 to Level 1 would be useful here.
The important change is that the global write now reads from
`shared[threadIdx.x][threadIdx.y]`, while the tile itself lands in the swapped
block position.




### Interlude 2: what TMA changes

![Figure19](Screenshot%20from%202026-06-04%2018-54-03.png)

The next version is where the problem changes character. The shared-memory
version still asks many threads to compute addresses and move data into a tile.
With TMA, one elected thread describes the tile, and the hardware copy engine
moves it asynchronously.

cited from Nvidia's H100 White paper(pp33 - 34):

The TMA operation is asynchronous and leverages the shared memory-based
asynchronous barriers introduced in A100. Additionally, the TMA programming
model is single-threaded, where a single thread in a warp is elected to issue an
asynchronous TMA operation (```cuda::memcpy_async```) to copy a tensor, and
subsequently multiple threads can wait on a ```cuda::barrier``` for completion
of the data transfer.

To further improve performance, the H100 SM adds hardware to accelerate these
asynchronous barrier wait operations. A key advantage of TMA is it frees the
threads to execute other independent work. On A100, in the left part of Figure
19, asynchronous memory copies were executed using a special
```LoadGlobalStoreShared``` instruction, so the threads were responsible for
generating all addresses and looping across the whole copy region.

On Hopper, TMA takes care of everything. A single thread creates a copy
descriptor before launching the TMA, and from then on address generation and
data movement are handled in hardware. TMA provides a much simpler programming
model because it takes over the task of computing stride, offset, and boundary
calculations when copying segments of a tensor.

At least, that is the nice version. The part that took me a while is that TMA is
not a drop-in replacement for pointer arithmetic. With normal CUDA, if I can
write the right address expression, I can probably load the data. With TMA, I
need to describe a legal tensor layout to the hardware.

### Level 3: Hopper TMA transpose (WIP)

For Level 1, shared memory was just a local staging tile. For Level 3, the tile
is still there, but the load into that tile is issued through TMA.

The conceptual pipeline becomes:

```
one elected thread describes the input tile
TMA asynchronously loads the tile into shared memory
the block waits on a transaction-aware barrier
threads write the transposed tile to global memory
```

The actual code is not the hard part to read. The hard part is understanding
what assumptions the descriptor is making. These are the pitfalls I hit.

#### Pitfall 1: TMA is initiated by one thread, not all threads

A TMA copy is usually issued by one elected thread, often:
```
if (threadIdx.x == 0 && threadIdx.y == 0) {
    cp_async_bulk_tensor(...);
}
```

My wrong version was:

```
if (threadIdx.x == 0) {
    cp_async_bulk_tensor(...);
}
```

The rest of the block waits on a barrier. This feels weird at first because
normal CUDA tiled loads are usually "every thread loads one element." With TMA,
the hardware copy engine moves the whole tile.

#### Pitfall 2: CUtensorMap is not a pointer

This was a big one. We cannot do:
```
output_desc[i] = ...
```

A tensor map is only a descriptor that tells TMA how to interpret global memory.
It is not the actual data buffer. For stores, we either pass the real `float*`
output, or later learn the TMA-store API separately.

#### Pitfall 3: descriptors care about physical layout, not just logical shape

Our logical matrix can be 2 x 3, but if it is row-major contiguous, the physical
row stride is:
```
cols * sizeof(float) = 3 * 4 = 12 bytes
```

For 2D TMA, `global_stride` has alignment constraints. A 12-byte row stride is
**illegal**. So we need a padded physical layout:
```
[1 2 3 pad]
[4 5 6 pad]
```
while the logical shape remains 2 x 3.

This is why the code later creates `padded_input` and uses `cudaMemcpy2D`.

#### Pitfall 4: global_dim, global_stride, and coords are in TMA tensor order

For a row-major matrix, it is easiest to think:
```
x = col
y = row
```
So:
```
global_dim = {cols, rows};
coords     = {tile_col_start, tile_row_start};
```
This is very easy to reverse by accident.

#### Pitfall 5: shared-memory padding and TMA layout are different things

For a normal shared-memory transpose, people often write:
```float tile[32][33];```to avoid bank conflicts. But TMA copies a dense box into shared memory according
to the descriptor and box dimensions. Randomly using `[TILE][TILE + 1]` can confuse the issue. First get `float tile[32][32]` working. Then think about
swizzle, padding, and bank conflicts.

#### Pitfall 6: the PTX wrapper overload is picky

The correct overload depends on the CUDA version and direction. In my case, the
compiler wanted:
```
cuda::ptx::space_cluster,
cuda::ptx::space_global,
dst,
tensor_map,
coords,
barrier_native_handle
```
not `space_shared`, and not `&bar` directly.

So this part matters:

```
cuda::device::barrier_native_handle(bar)
```

#### Pitfall 7: TMA needs transaction-aware barriers

A normal `__syncthreads()` is not enough to wait for the async TMA copy. We need something like:

```
token = cuda::device::barrier_arrive_tx(bar, 1, bytes);
bar.wait(std::move(token));
```

The copy is asynchronous, so the barrier must know there is an outstanding
memory transaction.

#### Pitfall 8: barrier_arrive_tx(..., bytes) should match the TMA transaction size

For a full 32 x 32 float tile:

```
32 * 32 * 4 = 4096 bytes
```

But if the TMA box is smaller, the byte count should not blindly be
`sizeof(tile)` unless I know the descriptor always copies the full tile.
Otherwise synchronization can get subtle.

#### Pitfall 9: edge tiles are annoying

If matrix dimensions are not multiples of the tile size, the tile at the
boundary may partially go out of bounds. We need to understand the OOB fill
mode:

```
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
```

This means do not rely on out-of-bounds elements being zero. The final store
guard still matters:

```
if (in_row < rows && in_col < cols) {
    ...
}
```

#### Pitfall 10: TMA requires Hopper-specific compilation

We generally want: ```nvcc -std=c++20 -arch=sm_90a ...```
Not just generic CUDA compilation. TMA is a Hopper feature, so compile target
matters.

#### Pitfall 11: descriptor creation can fail before the kernel even launches

This is what happened with my 2 x 3 matrix. The program crashed in:

```
cuTensorMapEncodeTiled(...)
```

not inside the kernel. So when debugging TMA, always print the `CUresult`
instead of just asserting.

#### Pitfall 12: TMA is descriptor-driven

This is the meta-pitfall. With normal CUDA, we can load arbitrary addresses as
long as the indexing is correct. With TMA, the data must be expressible as a
tensor with legal dimensions, legal strides, legal alignment, legal box size,
and compatible shared-memory layout. It is more rigid, but that rigidity is
what lets the hardware move large tiles efficiently.

The three lessons I want to keep in my head are:

1. TMA feels single-threaded at the issuing side.
2. Physical layout and padding suddenly matter.
3. TMA is descriptor-driven. I am no longer just writing pointer arithmetic; I
   am describing a legal tensor layout to hardware.

### Interlude 3: SWIZZLE keyword and bank conflict

First off, what is bank conflict?

And then, what is the SWIZZLE keyword?(we included in the TMA descriptor)


passed on B300


```
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cassert>
#include <cstdio>

#define TILE 32

using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__global__ void matrix_transpose_kernel(
    const __grid_constant__ CUtensorMap input_desc,
    float* __restrict__ output,
    int rows,
    int cols
) {
    __shared__ barrier_t bar;
    __shared__ alignas(128) float tile[TILE][TILE];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        init(&bar, blockDim.x * blockDim.y);
    }
    __syncthreads();

    barrier_t::arrival_token token;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int32_t coords[2] = {
            (int32_t)(blockIdx.x * TILE), // col
            (int32_t)(blockIdx.y * TILE)  // row
        };

        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_cluster,
            cuda::ptx::space_global,
            &tile[0][0],
            &input_desc,
            coords,
            cuda::device::barrier_native_handle(bar)
        );

        token = cuda::device::barrier_arrive_tx(
            bar,
            1,
            sizeof(tile)
        );
    } else {
        token = bar.arrive();
    }

    bar.wait(std::move(token));

    int in_col = blockIdx.x * TILE + threadIdx.x;
    int in_row = blockIdx.y * TILE + threadIdx.y;

    if (in_row < rows && in_col < cols) {
        int out_row = in_col;
        int out_col = in_row;
        // transpose here
        output[out_row * rows + out_col] = tile[threadIdx.y][threadIdx.x];
    }
}


CUtensorMap make_tma_desc_pitched(
    void* ptr,
    int rows,
    int cols,
    int ld_cols
) {
    CUtensorMap desc{};
    constexpr uint32_t rank = 2;

    // TMA tensor order: x = col, y = row.
    uint64_t global_dim[rank] = {
        (uint64_t)cols,
        (uint64_t)rows
    };

    // Row stride in bytes. Must be 16-byte aligned.
    uint64_t global_stride[rank - 1] = {
        (uint64_t)ld_cols * sizeof(float)
    };

    uint32_t box_dim[rank] = {
        (uint32_t)((cols < TILE) ? cols : TILE),
        (uint32_t)((rows < TILE) ? rows : TILE)
    };

    uint32_t elem_stride[rank] = {
        1,
        1
    };

    CUresult res = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
        rank,
        ptr,
        global_dim,
        global_stride,
        box_dim,
        elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (res != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* msg = nullptr;
        cuGetErrorName(res, &name);
        cuGetErrorString(res, &msg);
        printf("cuTensorMapEncodeTiled failed: %s: %s\n", name, msg);
        assert(false);
    }

    return desc;
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    // TMA 2D descriptor needs row stride multiple of 16 bytes.
    // For float, that means leading dimension must be multiple of 4.
    int ld_cols = ((cols + 3) / 4) * 4;

    float* padded_input = nullptr;
    cudaMalloc(&padded_input, (size_t)rows * ld_cols * sizeof(float));

    // Optional but useful: padding becomes deterministic.
    cudaMemset(padded_input, 0, (size_t)rows * ld_cols * sizeof(float));

    // Copy original row-major input into padded row-major input.
    cudaMemcpy2D(
        padded_input,
        (size_t)ld_cols * sizeof(float),
        input,
        (size_t)cols * sizeof(float),
        (size_t)cols * sizeof(float),
        rows,
        cudaMemcpyDeviceToDevice
    );

    CUtensorMap input_desc =
        make_tma_desc_pitched(padded_input, rows, cols, ld_cols);

    dim3 threadsPerBlock(TILE, TILE);
    dim3 blocksPerGrid(
        (cols + TILE - 1) / TILE,
        (rows + TILE - 1) / TILE
    );

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input_desc,
        output,
        rows,
        cols
    );

    cudaDeviceSynchronize();

    cudaFree(padded_input);
}

```
Driver portion:

```
import ctypes
from typing import Any, Dict, List

import torch
from core.challenge_base import ChallengeBase


class Challenge(ChallengeBase):
    name = "Matrix Transpose"
    atol = 1e-05
    rtol = 1e-05
    num_gpus = 1
    access_tier = "free"

    def reference_impl(self, input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
        assert input.shape == (rows, cols)
        assert output.shape == (cols, rows)
        assert input.dtype == output.dtype
        assert input.device == output.device

        output.copy_(input.transpose(0, 1))

    def get_solve_signature(self) -> Dict[str, tuple]:
        return {
            "input": (ctypes.POINTER(ctypes.c_float), "in"),
            "output": (ctypes.POINTER(ctypes.c_float), "out"),
            "rows": (ctypes.c_int, "in"),
            "cols": (ctypes.c_int, "in"),
        }

    def generate_example_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 2, 3
        input_tensor = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=self.device, dtype=dtype
        )
        output_tensor = torch.empty(cols, rows, device=self.device, dtype=dtype)
        return {
            "input": input_tensor,
            "output": output_tensor,
            "rows": rows,
            "cols": cols,
        }

    def generate_functional_test(self) -> List[Dict[str, Any]]:
        dtype = torch.float32
        test_specs = [
            # Basic test cases
            ("basic_2x3", 2, 3, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            ("basic_3x1", 3, 1, [[1.0], [2.0], [3.0]]),
            ("square_2x2", 2, 2, [[1.0, 2.0], [3.0, 4.0]]),
            ("single_row", 1, 4, [[1.0, 2.0, 3.0, 4.0]]),
            ("single_column", 4, 1, [[1.0], [2.0], [3.0], [4.0]]),
        ]

        test_cases = []
        for _, r, c, input_vals in test_specs:
            test_cases.append(
                {
                    "input": torch.tensor(input_vals, device=self.device, dtype=dtype),
                    "output": torch.empty(c, r, device=self.device, dtype=dtype),
                    "rows": r,
                    "cols": c,
                }
            )

        # Random test cases with different sizes
        for _, rows, cols in [
            ("small_rectangular", 4, 6),
            ("medium_square", 8, 8),
            ("large_rectangular", 16, 12),
            ("tall_matrix", 32, 8),
            ("wide_matrix", 8, 32),
        ]:
            test_cases.append(
                {
                    "input": torch.empty(rows, cols, device=self.device, dtype=dtype).uniform_(
                        -10.0, 10.0
                    ),
                    "output": torch.empty(cols, rows, device=self.device, dtype=dtype),
                    "rows": rows,
                    "cols": cols,
                }
            )

        # Edge cases
        for _, rows, cols in [
            ("single_element", 1, 1),
            ("max_dimensions", 8192, 8192),
        ]:
            test_cases.append(
                {
                    "input": torch.empty(rows, cols, device=self.device, dtype=dtype).uniform_(
                        -1.0, 1.0
                    ),
                    "output": torch.empty(cols, rows, device=self.device, dtype=dtype),
                    "rows": rows,
                    "cols": cols,
                }
            )

        return test_cases

    def generate_performance_test(self) -> Dict[str, Any]:
        dtype = torch.float32
        rows, cols = 7000, 6000
        return {
            "input": torch.empty(rows, cols, device=self.device, dtype=dtype).uniform_(-10.0, 10.0),
            "output": torch.zeros(cols, rows, device=self.device, dtype=dtype),
            "rows": rows,
            "cols": cols,
        }

```
Assume FP32 matrix, 
compiled with sm_90 and got
```
64x64: PASS max_err 0.0
32x32: PASS max_err 0.0
33x33: PASS max_err 0.0
127x251: PASS max_err 0.0
1024x1024: PASS max_err 0.0
7000x6000: PASS max_err 0.0 elapsed_ms 16.818
```

compiled with sm_100 and got
```
64x64: PASS max_err 0.0 elapsed_ms 17.986
33x33: PASS max_err 0.0 elapsed_ms 0.802
127x251: PASS max_err 0.0 elapsed_ms 0.520
1024x1024: PASS max_err 0.0 elapsed_ms 0.495
7000x6000: PASS max_err 0.0 elapsed_ms 1.046
```


test harness:
(base) uccl@mi-sky-b300:~/yuyi$ python - <<'PY'
import ctypes
import time
import torch

so = "/home/uccl/yuyi/leetgpu-challenges/challenges/easy/3_matrix_transpose/build/starter.so"

lib = ctypes.CDLL(so)
lib.solve.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.solve.restype = None

torch.cuda.set_device(0)

cases = [(64, 64), (33, 33), (127, 251), (1024, 1024), (7000, 6000)]

for rows, cols in cases:
    x = torch.empty((rows, cols), device="cuda", dtype=torch.float32).uniform_(-10, 10)
    y = torch.empty((cols, rows), device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    start = time.perf_counter()

    lib.solve(x.data_ptr(), y.data_ptr(), rows, cols)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    ref = x.t().contiguous()
    ok = torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
    max_err = (y - ref).abs().max().item()

PY  print(f"{rows}x{cols}: {'PASS' if ok else 'FAIL'} max_err {max_err} elapsed_ms {elapsed_ms:.3f}")





Blackwell specific optimization

tcgen05, which is the Tensor Cores in the latest Nvidia Blackwell series GPUs


Reference:

https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf
https://research.colfax-intl.com/tutorial-hopper-tma/
https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf
https://gau-nernst.github.io/tcgen05/
