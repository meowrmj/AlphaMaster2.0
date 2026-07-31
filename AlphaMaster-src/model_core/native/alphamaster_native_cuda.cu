#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {

constexpr int OP_ADD = 1;
constexpr int OP_SUB = 2;
constexpr int OP_MUL = 3;
constexpr int OP_DIV = 4;
constexpr int OP_MAX = 5;
constexpr int OP_MIN = 6;
constexpr int OP_IF_GT = 101;
constexpr int OP_GATE = 102;

template <typename scalar_t>
__device__ __forceinline__ scalar_t sanitize(scalar_t x) {
  return isfinite(static_cast<double>(x)) ? x : static_cast<scalar_t>(0);
}

template <typename scalar_t>
__global__ void elementwise2_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  scalar_t av = a[i];
  scalar_t bv = b[i];
  scalar_t v = static_cast<scalar_t>(0);
  if (op_id == OP_ADD) {
    v = av + bv;
  } else if (op_id == OP_SUB) {
    v = av - bv;
  } else if (op_id == OP_MUL) {
    v = av * bv;
  } else if (op_id == OP_DIV) {
    v = av / (bv + static_cast<scalar_t>(1e-6));
  } else if (op_id == OP_MAX) {
    v = av > bv ? av : bv;
  } else if (op_id == OP_MIN) {
    v = av < bv ? av : bv;
  }
  out[i] = sanitize(v);
}

template <typename scalar_t>
__global__ void elementwise3_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  scalar_t av = a[i];
  scalar_t bv = b[i];
  scalar_t cv = c[i];
  scalar_t v = static_cast<scalar_t>(0);
  if (op_id == OP_IF_GT) {
    v = av > static_cast<scalar_t>(0) ? bv : cv;
  } else if (op_id == OP_GATE) {
    scalar_t mask = av > static_cast<scalar_t>(0) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0);
    v = mask * bv + (static_cast<scalar_t>(1) - mask) * cv;
  }
  out[i] = sanitize(v);
}

}  // namespace

torch::Tensor elementwise2_cuda(torch::Tensor a, torch::Tensor b, int64_t op_id) {
  auto out = torch::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "elementwise2_cuda", [&] {
    elementwise2_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        n,
        op_id);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor elementwise3_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t op_id) {
  auto out = torch::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "elementwise3_cuda", [&] {
    elementwise3_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        c.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        n,
        op_id);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
