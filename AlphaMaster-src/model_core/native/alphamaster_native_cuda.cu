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
constexpr int OP_NEG = 51;
constexpr int OP_ABS = 52;
constexpr int OP_SIGN = 53;
constexpr int OP_POWER = 54;
constexpr int OP_SIGNED_POWER_2 = 55;
constexpr int OP_SIGNED_LOG = 56;
constexpr int OP_SQRT = 57;
constexpr int OP_CLIP = 58;
constexpr int OP_SIGMOID = 59;
constexpr int OP_TANH_SQUASH = 60;
constexpr int OP_DELAY1 = 201;
constexpr int OP_DELAY4 = 204;
constexpr int OP_DELTA = 211;
constexpr int OP_DELTA_5 = 215;
constexpr int OP_TS_MEAN_5 = 305;
constexpr int OP_TS_MEAN_10 = 310;
constexpr int OP_TS_MEAN_20 = 320;
constexpr int OP_TS_SUM_5 = 405;
constexpr int OP_TS_SUM_10 = 410;
constexpr int OP_TS_SUM_20 = 420;
constexpr int OP_TS_ZSCORE_10 = 510;
constexpr int OP_TS_ZSCORE_20 = 520;

template <typename scalar_t>
__device__ __forceinline__ scalar_t sanitize(scalar_t x) {
  return isfinite(static_cast<double>(x)) ? x : static_cast<scalar_t>(0);
}

template <typename scalar_t>
__global__ void elementwise1_kernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  scalar_t av = a[i];
  scalar_t v = static_cast<scalar_t>(0);
  if (op_id == OP_NEG) {
    v = -av;
  } else if (op_id == OP_ABS) {
    v = av < static_cast<scalar_t>(0) ? -av : av;
  } else if (op_id == OP_SIGN) {
    v = (av > static_cast<scalar_t>(0)) - (av < static_cast<scalar_t>(0));
  } else if (op_id == OP_POWER || op_id == OP_SIGNED_POWER_2) {
    scalar_t abs_v = av < static_cast<scalar_t>(0) ? -av : av;
    scalar_t sign_v = (av > static_cast<scalar_t>(0)) - (av < static_cast<scalar_t>(0));
    v = sign_v * abs_v * abs_v;
  } else if (op_id == OP_SIGNED_LOG) {
    scalar_t abs_v = av < static_cast<scalar_t>(0) ? -av : av;
    scalar_t sign_v = (av > static_cast<scalar_t>(0)) - (av < static_cast<scalar_t>(0));
    v = sign_v * log1p(abs_v);
  } else if (op_id == OP_SQRT) {
    scalar_t abs_v = av < static_cast<scalar_t>(0) ? -av : av;
    scalar_t sign_v = (av > static_cast<scalar_t>(0)) - (av < static_cast<scalar_t>(0));
    v = sign_v * sqrt(abs_v);
  } else if (op_id == OP_CLIP) {
    v = av < static_cast<scalar_t>(-3) ? static_cast<scalar_t>(-3) :
        (av > static_cast<scalar_t>(3) ? static_cast<scalar_t>(3) : av);
  } else if (op_id == OP_SIGMOID) {
    v = static_cast<scalar_t>(2) / (static_cast<scalar_t>(1) + exp(-av)) - static_cast<scalar_t>(1);
  } else if (op_id == OP_TANH_SQUASH) {
    v = tanh(av);
  }
  out[i] = sanitize(v);
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

__device__ __forceinline__ int window_for_op(int64_t op_id) {
  if (op_id == OP_TS_MEAN_5 || op_id == OP_TS_SUM_5) {
    return 5;
  }
  if (op_id == OP_TS_MEAN_10 || op_id == OP_TS_SUM_10 || op_id == OP_TS_ZSCORE_10) {
    return 10;
  }
  if (op_id == OP_TS_MEAN_20 || op_id == OP_TS_SUM_20 || op_id == OP_TS_ZSCORE_20) {
    return 20;
  }
  return 1;
}

template <typename scalar_t>
__global__ void shift1_kernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t n_symbols,
    int64_t n_bars,
    int64_t op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = bsz * n_symbols * n_bars;
  if (i >= total) {
    return;
  }
  int64_t t = i % n_bars;
  int delay = 1;
  if (op_id == OP_DELAY4 || op_id == OP_DELTA_5) {
    delay = (op_id == OP_DELAY4) ? 4 : 5;
  }
  scalar_t shifted = t >= delay ? a[i - delay] : static_cast<scalar_t>(0);
  scalar_t v = shifted;
  if (op_id == OP_DELTA || op_id == OP_DELTA_5) {
    v = a[i] - shifted;
  }
  out[i] = sanitize(v);
}

template <typename scalar_t>
__global__ void rolling1_kernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t n_symbols,
    int64_t n_bars,
    int64_t op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = bsz * n_symbols * n_bars;
  if (i >= total) {
    return;
  }
  int64_t t = i % n_bars;
  int w = window_for_op(op_id);
  scalar_t sum = static_cast<scalar_t>(0);
  scalar_t sum_sq = static_cast<scalar_t>(0);
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t v = src_t >= 0 ? a[i + (src_t - t)] : static_cast<scalar_t>(0);
    sum += v;
    sum_sq += v * v;
  }
  scalar_t out_v = sum;
  if (op_id == OP_TS_MEAN_5 || op_id == OP_TS_MEAN_10 || op_id == OP_TS_MEAN_20) {
    out_v = sum / static_cast<scalar_t>(w);
  } else if (op_id == OP_TS_ZSCORE_10 || op_id == OP_TS_ZSCORE_20) {
    scalar_t mean = sum / static_cast<scalar_t>(w);
    scalar_t var = sum_sq / static_cast<scalar_t>(w) - mean * mean;
    scalar_t std_v = sqrt(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0));
    out_v = (a[i] - mean) / (std_v + static_cast<scalar_t>(1e-6));
  }
  out[i] = sanitize(out_v);
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

torch::Tensor elementwise1_cuda(torch::Tensor a, int64_t op_id) {
  auto out = torch::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "elementwise1_cuda", [&] {
    elementwise1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        n,
        op_id);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

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

torch::Tensor shift1_cuda(torch::Tensor a, int64_t op_id) {
  auto out = torch::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "shift1_cuda", [&] {
    shift1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        bsz,
        n_symbols,
        n_bars,
        op_id);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor rolling1_cuda(torch::Tensor a, int64_t op_id) {
  auto out = torch::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "rolling1_cuda", [&] {
    rolling1_kernel<scalar_t><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(),
        bsz,
        n_symbols,
        n_bars,
        op_id);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
