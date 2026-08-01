#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
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
constexpr int OP_TS_STD_5 = 605;
constexpr int OP_TS_STD_10 = 610;
constexpr int OP_TS_STD_20 = 620;
constexpr int OP_TS_RANK_5 = 705;
constexpr int OP_TS_RANK_10 = 710;
constexpr int OP_TS_RANK_20 = 720;
constexpr int OP_TS_MIN_10 = 810;
constexpr int OP_TS_MIN_20 = 820;
constexpr int OP_TS_MAX_10 = 910;
constexpr int OP_TS_MAX_20 = 920;
constexpr int OP_TS_QUANTILE_10 = 1010;
constexpr int OP_TS_ARG_MAX_5 = 1105;
constexpr int OP_TS_ARG_MIN_5 = 1205;
constexpr int OP_DECAY = 1303;
constexpr int OP_WMA = 1304;
constexpr int OP_DECAY_LINEAR_5 = 1305;
constexpr int OP_TS_DECAY_EXP_5 = 1306;
constexpr int OP_EMA_5 = 1405;
constexpr int OP_EMA_20 = 1420;
constexpr int OP_MOMENTUM_5 = 1505;
constexpr int OP_MOMENTUM_10 = 1510;
constexpr int OP_MAX3 = 1603;

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
  if (op_id == OP_TS_MEAN_10 || op_id == OP_TS_SUM_10 || op_id == OP_TS_ZSCORE_10 ||
      op_id == OP_TS_STD_10 || op_id == OP_TS_RANK_10 || op_id == OP_TS_MIN_10 ||
      op_id == OP_TS_MAX_10 || op_id == OP_TS_QUANTILE_10) {
    return 10;
  }
  if (op_id == OP_TS_MEAN_20 || op_id == OP_TS_SUM_20 || op_id == OP_TS_ZSCORE_20 ||
      op_id == OP_TS_STD_20 || op_id == OP_TS_RANK_20 || op_id == OP_TS_MIN_20 ||
      op_id == OP_TS_MAX_20) {
    return 20;
  }
  if (op_id == OP_TS_STD_5 || op_id == OP_TS_RANK_5 || op_id == OP_TS_ARG_MAX_5 ||
      op_id == OP_TS_ARG_MIN_5 || op_id == OP_DECAY_LINEAR_5 || op_id == OP_TS_DECAY_EXP_5) {
    return 5;
  }
  return 1;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t value_at(
    const scalar_t* __restrict__ a,
    int64_t base,
    int64_t t,
    int64_t n_bars) {
  return (t >= 0 && t < n_bars) ? a[base + t] : static_cast<scalar_t>(0);
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
  scalar_t cur = a[i];
  scalar_t min_v = static_cast<scalar_t>(0);
  scalar_t max_v = static_cast<scalar_t>(0);
  int arg_min = 0;
  int arg_max = 0;
  int rank_count = 0;
  bool first = true;
  int64_t base = i - t;
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t v = value_at(a, base, src_t, n_bars);
    sum += v;
    sum_sq += v * v;
    if (first || v < min_v) {
      min_v = v;
      arg_min = k;
    }
    if (first || v > max_v) {
      max_v = v;
      arg_max = k;
    }
    if (v < cur) {
      rank_count += 1;
    }
    first = false;
  }
  scalar_t out_v = sum;
  if (op_id == OP_TS_MEAN_5 || op_id == OP_TS_MEAN_10 || op_id == OP_TS_MEAN_20) {
    out_v = sum / static_cast<scalar_t>(w);
  } else if (op_id == OP_TS_STD_5 || op_id == OP_TS_STD_10 || op_id == OP_TS_STD_20) {
    scalar_t mean = sum / static_cast<scalar_t>(w);
    scalar_t var = sum_sq / static_cast<scalar_t>(w) - mean * mean;
    out_v = sqrt(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0)) + static_cast<scalar_t>(1e-6);
  } else if (op_id == OP_TS_ZSCORE_10 || op_id == OP_TS_ZSCORE_20) {
    scalar_t mean = sum / static_cast<scalar_t>(w);
    scalar_t var = sum_sq / static_cast<scalar_t>(w) - mean * mean;
    scalar_t std_v = sqrt(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0));
    out_v = (cur - mean) / (std_v + static_cast<scalar_t>(1e-6));
  } else if (op_id == OP_TS_RANK_5 || op_id == OP_TS_RANK_10 || op_id == OP_TS_RANK_20 ||
             op_id == OP_TS_QUANTILE_10) {
    out_v = static_cast<scalar_t>(rank_count) / static_cast<scalar_t>(w);
  } else if (op_id == OP_TS_MIN_10 || op_id == OP_TS_MIN_20) {
    out_v = min_v;
  } else if (op_id == OP_TS_MAX_10 || op_id == OP_TS_MAX_20) {
    out_v = max_v;
  } else if (op_id == OP_TS_ARG_MAX_5) {
    out_v = static_cast<scalar_t>(arg_max) / static_cast<scalar_t>(w - 1);
  } else if (op_id == OP_TS_ARG_MIN_5) {
    out_v = static_cast<scalar_t>(arg_min) / static_cast<scalar_t>(w - 1);
  } else if (op_id == OP_DECAY) {
    out_v = (cur + static_cast<scalar_t>(0.8) * value_at(a, base, t - 1, n_bars) +
             static_cast<scalar_t>(0.6) * value_at(a, base, t - 2, n_bars)) / static_cast<scalar_t>(2.4);
  } else if (op_id == OP_WMA) {
    out_v = (static_cast<scalar_t>(3.0) * cur +
             static_cast<scalar_t>(2.0) * value_at(a, base, t - 1, n_bars) +
             value_at(a, base, t - 2, n_bars)) / static_cast<scalar_t>(6.0);
  } else if (op_id == OP_DECAY_LINEAR_5) {
    out_v = static_cast<scalar_t>(0);
    for (int k = 0; k < 5; ++k) {
      int64_t src_t = t - (4 - k);
      out_v += value_at(a, base, src_t, n_bars) * static_cast<scalar_t>(k + 1);
    }
    out_v = out_v / static_cast<scalar_t>(15.0);
  } else if (op_id == OP_TS_DECAY_EXP_5) {
    const scalar_t weights[5] = {
        static_cast<scalar_t>(0.0322580645),
        static_cast<scalar_t>(0.0645161290),
        static_cast<scalar_t>(0.1290322581),
        static_cast<scalar_t>(0.2580645161),
        static_cast<scalar_t>(0.5161290323)};
    out_v = static_cast<scalar_t>(0);
    for (int k = 0; k < 5; ++k) {
      int64_t src_t = t - (4 - k);
      out_v += value_at(a, base, src_t, n_bars) * weights[k];
    }
  } else if (op_id == OP_MOMENTUM_5 || op_id == OP_MOMENTUM_10) {
    int short_w = (op_id == OP_MOMENTUM_5) ? 5 : 10;
    scalar_t short_sum = static_cast<scalar_t>(0);
    scalar_t long_sum = static_cast<scalar_t>(0);
    for (int k = 0; k < short_w; ++k) {
      short_sum += value_at(a, base, t - (short_w - 1 - k), n_bars);
    }
    for (int k = 0; k < 20; ++k) {
      long_sum += value_at(a, base, t - (19 - k), n_bars);
    }
    out_v = short_sum / static_cast<scalar_t>(short_w) - long_sum / static_cast<scalar_t>(20);
  } else if (op_id == OP_MAX3) {
    scalar_t d1 = value_at(a, base, t - 1, n_bars);
    scalar_t d2 = value_at(a, base, t - 2, n_bars);
    out_v = cur > d1 ? cur : d1;
    out_v = out_v > d2 ? out_v : d2;
  }
  out[i] = sanitize(out_v);
}

template <typename scalar_t>
__global__ void ema1_kernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ out,
    int64_t series_count,
    int64_t n_bars,
    int64_t op_id) {
  int64_t series = blockIdx.x * blockDim.x + threadIdx.x;
  if (series >= series_count) {
    return;
  }
  scalar_t alpha = op_id == OP_EMA_5 ? static_cast<scalar_t>(2.0 / 6.0) : static_cast<scalar_t>(2.0 / 21.0);
  int64_t base = series * n_bars;
  if (n_bars <= 0) {
    return;
  }
  scalar_t prev = a[base];
  out[base] = sanitize(prev);
  for (int64_t t = 1; t < n_bars; ++t) {
    scalar_t cur = a[base + t];
    prev = alpha * cur + (static_cast<scalar_t>(1) - alpha) * prev;
    out[base + t] = sanitize(prev);
  }
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

at::Tensor elementwise1_cuda(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  elementwise1_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      out.data_ptr<float>(),
      n,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor elementwise2_cuda(at::Tensor a, at::Tensor b, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  elementwise2_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      n,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor elementwise3_cuda(at::Tensor a, at::Tensor b, at::Tensor c, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  elementwise3_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      out.data_ptr<float>(),
      n,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor shift1_cuda(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  shift1_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor rolling1_cuda(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  if (op_id == OP_EMA_5 || op_id == OP_EMA_20) {
    int64_t series_count = bsz * n_symbols;
    constexpr int threads = 128;
    int blocks = static_cast<int>((series_count + threads - 1) / threads);
    ema1_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        a.data_ptr<float>(),
        out.data_ptr<float>(),
        series_count,
        n_bars,
        op_id);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  rolling1_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
