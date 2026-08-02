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
constexpr int OP_WINSORIZE = 580;
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
constexpr int OP_TS_SKEW_10 = 1020;
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
constexpr int OP_TS_CORR_10 = 2010;
constexpr int OP_COVARIANCE_10 = 2020;
constexpr int OP_CS_SCALE = 3020;
constexpr int OP_CS_NEUTRALIZE = 3030;

template <typename scalar_t>
__device__ __forceinline__ scalar_t sanitize(scalar_t x) {
  return isfinite(static_cast<double>(x)) ? x : static_cast<scalar_t>(0);
}

__device__ __forceinline__ float add_rn(float a, float b) {
  return __fadd_rn(a, b);
}

__device__ __forceinline__ float mul_rn(float a, float b) {
  return __fmul_rn(a, b);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_unary_op(scalar_t av, int64_t op_id) {
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
    v = sign_v * sqrtf(abs_v);
  } else if (op_id == OP_CLIP) {
    v = av < static_cast<scalar_t>(-3) ? static_cast<scalar_t>(-3) :
        (av > static_cast<scalar_t>(3) ? static_cast<scalar_t>(3) : av);
  } else if (op_id == OP_SIGMOID) {
    v = static_cast<scalar_t>(2) / (static_cast<scalar_t>(1) + exp(-av)) - static_cast<scalar_t>(1);
  } else if (op_id == OP_TANH_SQUASH) {
    v = tanh(av);
  }
  return sanitize(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_binary_op(scalar_t av, scalar_t bv, int64_t op_id) {
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
  return sanitize(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_branch_op(
    scalar_t av,
    scalar_t bv,
    scalar_t cv,
    int64_t op_id) {
  scalar_t v = static_cast<scalar_t>(0);
  if (op_id == OP_IF_GT) {
    v = av > static_cast<scalar_t>(0) ? bv : cv;
  } else if (op_id == OP_GATE) {
    scalar_t mask = av > static_cast<scalar_t>(0) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(0);
    v = mask * bv + (static_cast<scalar_t>(1) - mask) * cv;
  }
  return sanitize(v);
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
  out[i] = apply_unary_op(a[i], op_id);
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
  out[i] = apply_binary_op(av, bv, op_id);
}

__device__ __forceinline__ int window_for_op(int64_t op_id) {
  if (op_id == OP_TS_MEAN_5 || op_id == OP_TS_SUM_5) {
    return 5;
  }
  if (op_id == OP_TS_MEAN_10 || op_id == OP_TS_SUM_10 || op_id == OP_TS_ZSCORE_10 ||
      op_id == OP_TS_STD_10 || op_id == OP_TS_RANK_10 || op_id == OP_TS_MIN_10 ||
      op_id == OP_TS_MAX_10 || op_id == OP_TS_QUANTILE_10 || op_id == OP_TS_SKEW_10) {
    return 10;
  }
  if (op_id == OP_TS_MEAN_20 || op_id == OP_TS_SUM_20 || op_id == OP_TS_ZSCORE_20 ||
      op_id == OP_TS_STD_20 || op_id == OP_TS_RANK_20 || op_id == OP_TS_MIN_20 ||
      op_id == OP_TS_MAX_20 || op_id == OP_WINSORIZE) {
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
__device__ __forceinline__ scalar_t apply_rolling_max_op(
    const scalar_t* __restrict__ a,
    int64_t flat_i,
    int64_t t,
    int64_t n_bars,
    int64_t op_id) {
  int w = op_id == OP_TS_MAX_20 ? 20 : 10;
  int64_t base = flat_i - t;
  scalar_t max_v = static_cast<scalar_t>(0);
  bool first = true;
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t v = value_at(a, base, src_t, n_bars);
    if (first || v > max_v) {
      max_v = v;
    }
    first = false;
  }
  return sanitize(max_v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_first_unary_at(
    const scalar_t* __restrict__ a,
    int64_t base,
    int64_t t,
    int64_t n_bars,
    int64_t op_id) {
  scalar_t cur = value_at(a, base, t, n_bars);
  if (op_id == OP_MAX3) {
    scalar_t d1 = value_at(a, base, t - 1, n_bars);
    scalar_t d2 = value_at(a, base, t - 2, n_bars);
    scalar_t out_v = cur > d1 ? cur : d1;
    return sanitize(out_v > d2 ? out_v : d2);
  }
  return apply_unary_op(cur, op_id);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t apply_second_unary_window_op(
    const scalar_t* __restrict__ a,
    int64_t flat_i,
    int64_t t,
    int64_t n_bars,
    int64_t first_unary_op_id,
    int64_t second_unary_op_id) {
  if (second_unary_op_id != OP_TS_ZSCORE_10 && second_unary_op_id != OP_TS_ZSCORE_20) {
    return apply_unary_op(apply_first_unary_at(a, flat_i - t, t, n_bars, first_unary_op_id), second_unary_op_id);
  }
  int w = second_unary_op_id == OP_TS_ZSCORE_20 ? 20 : 10;
  int64_t base = flat_i - t;
  scalar_t sum = static_cast<scalar_t>(0);
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    sum = add_rn(sum, apply_first_unary_at(a, base, src_t, n_bars, first_unary_op_id));
  }
  scalar_t mean = sum / static_cast<scalar_t>(w);
  scalar_t var = static_cast<scalar_t>(0);
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t centered = apply_first_unary_at(a, base, src_t, n_bars, first_unary_op_id) - mean;
    var = add_rn(var, mul_rn(centered, centered));
  }
  var = var / static_cast<scalar_t>(w);
  scalar_t std_v = sqrtf(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0));
  if (std_v < static_cast<scalar_t>(1e-6)) {
    return static_cast<scalar_t>(0);
  }
  scalar_t cur = apply_first_unary_at(a, base, t, n_bars, first_unary_op_id);
  return sanitize((cur - mean) / (std_v + static_cast<scalar_t>(1e-6)));
}

template <typename scalar_t>
__device__ __forceinline__ void sort_window(scalar_t* values, int n) {
  for (int i = 1; i < n; ++i) {
    scalar_t key = values[i];
    int j = i - 1;
    while (j >= 0 && values[j] > key) {
      values[j + 1] = values[j];
      --j;
    }
    values[j + 1] = key;
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t quantile_linear_sorted(
    const scalar_t* values,
    int n,
    scalar_t q) {
  scalar_t pos = q * static_cast<scalar_t>(n - 1);
  int lo = static_cast<int>(floor(pos));
  int hi = lo + 1;
  if (hi >= n) {
    hi = n - 1;
  }
  scalar_t weight = pos - static_cast<scalar_t>(lo);
  return values[lo] * (static_cast<scalar_t>(1) - weight) + values[hi] * weight;
}

__device__ __forceinline__ float sum5_torch_like(const float* v) {
  float sum = add_rn(v[0], v[1]);
  sum = add_rn(sum, v[2]);
  sum = add_rn(sum, v[3]);
  return add_rn(sum, v[4]);
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
__global__ void fused_shift_unary_kernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t n_symbols,
    int64_t n_bars,
    int64_t shift_op_id,
    int64_t unary_op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = bsz * n_symbols * n_bars;
  if (i >= total) {
    return;
  }
  int64_t t = i % n_bars;
  int delay = 1;
  if (shift_op_id == OP_DELAY4 || shift_op_id == OP_DELTA_5) {
    delay = (shift_op_id == OP_DELAY4) ? 4 : 5;
  }
  scalar_t shifted = t >= delay ? a[i - delay] : static_cast<scalar_t>(0);
  scalar_t v = shifted;
  if (shift_op_id == OP_DELTA || shift_op_id == OP_DELTA_5) {
    v = a[i] - shifted;
  }
  out[i] = apply_unary_op(sanitize(v), unary_op_id);
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
  scalar_t window_values[20];
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t v = value_at(a, base, src_t, n_bars);
    window_values[k] = v;
    sum = add_rn(sum, v);
    sum_sq = add_rn(sum_sq, mul_rn(v, v));
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
    scalar_t mean = op_id == OP_TS_STD_5
        ? mul_rn(sum5_torch_like(window_values), static_cast<scalar_t>(0.20000000298023224))
        : sum / static_cast<scalar_t>(w);
    scalar_t var = static_cast<scalar_t>(0);
    if (op_id == OP_TS_STD_5) {
      scalar_t sq[5];
      for (int k = 0; k < 5; ++k) {
        scalar_t centered = window_values[k] - mean;
        sq[k] = mul_rn(centered, centered);
      }
      var = mul_rn(sum5_torch_like(sq), static_cast<scalar_t>(0.20000000298023224));
    } else {
      for (int k = 0; k < w; ++k) {
        scalar_t centered = window_values[k] - mean;
        var = add_rn(var, mul_rn(centered, centered));
      }
      var = var / static_cast<scalar_t>(w);
    }
    out_v = sqrtf(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0)) + static_cast<scalar_t>(1e-6);
  } else if (op_id == OP_TS_ZSCORE_10 || op_id == OP_TS_ZSCORE_20) {
    scalar_t mean = sum / static_cast<scalar_t>(w);
    scalar_t var = static_cast<scalar_t>(0);
    for (int k = 0; k < w; ++k) {
      int64_t src_t = t - (w - 1 - k);
      scalar_t centered = value_at(a, base, src_t, n_bars) - mean;
      var = add_rn(var, mul_rn(centered, centered));
    }
    var = var / static_cast<scalar_t>(w);
    scalar_t std_v = sqrtf(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0));
    out_v = std_v < static_cast<scalar_t>(1e-6)
        ? static_cast<scalar_t>(0)
        : (cur - mean) / (std_v + static_cast<scalar_t>(1e-6));
  } else if (op_id == OP_WINSORIZE) {
    scalar_t values[20];
    for (int k = 0; k < 20; ++k) {
      int64_t src_t = t - (19 - k);
      values[k] = value_at(a, base, src_t, n_bars);
    }
    sort_window(values, 20);
    scalar_t lower = quantile_linear_sorted(values, 20, static_cast<scalar_t>(0.05));
    scalar_t upper = quantile_linear_sorted(values, 20, static_cast<scalar_t>(0.95));
    scalar_t span = upper - lower;
    if (span < static_cast<scalar_t>(1e-9)) {
      lower = cur;
      upper = cur;
    }
    out_v = cur < lower ? lower : (cur > upper ? upper : cur);
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
    scalar_t numerator = static_cast<scalar_t>(3.0) * cur +
        static_cast<scalar_t>(2.0) * value_at(a, base, t - 1, n_bars);
    numerator = numerator + value_at(a, base, t - 2, n_bars);
    out_v = numerator * static_cast<scalar_t>(0.1666666716337204);
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
  } else if (op_id == OP_TS_SKEW_10) {
    scalar_t mean = sum / static_cast<scalar_t>(w);
    scalar_t var = static_cast<scalar_t>(0);
    for (int k = 0; k < 10; ++k) {
      int64_t src_t = t - (9 - k);
      scalar_t centered = value_at(a, base, src_t, n_bars) - mean;
      var = add_rn(var, mul_rn(centered, centered));
    }
    var = var / static_cast<scalar_t>(10);
    scalar_t std_v = sqrtf(var > static_cast<scalar_t>(0) ? var : static_cast<scalar_t>(0)) + static_cast<scalar_t>(1e-6);
    scalar_t skew_sum = static_cast<scalar_t>(0);
    for (int k = 0; k < 10; ++k) {
      int64_t src_t = t - (9 - k);
      scalar_t z = (value_at(a, base, src_t, n_bars) - mean) / std_v;
      skew_sum = add_rn(skew_sum, mul_rn(mul_rn(z, z), z));
    }
    out_v = skew_sum / static_cast<scalar_t>(10);
    if (out_v < static_cast<scalar_t>(-5.0)) {
      out_v = static_cast<scalar_t>(-5.0);
    } else if (out_v > static_cast<scalar_t>(5.0)) {
      out_v = static_cast<scalar_t>(5.0);
    }
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
__global__ void rolling2_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
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
  int64_t base = i - t;
  constexpr int w = 10;
  scalar_t sum_x = static_cast<scalar_t>(0);
  scalar_t sum_y = static_cast<scalar_t>(0);
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t x = value_at(a, base, src_t, n_bars);
    scalar_t y = value_at(b, base, src_t, n_bars);
    sum_x += x;
    sum_y += y;
  }
  scalar_t inv_w = static_cast<scalar_t>(1.0 / w);
  scalar_t mean_x = sum_x * inv_w;
  scalar_t mean_y = sum_y * inv_w;
  scalar_t cov = static_cast<scalar_t>(0);
  scalar_t var_x = static_cast<scalar_t>(0);
  scalar_t var_y = static_cast<scalar_t>(0);
  for (int k = 0; k < w; ++k) {
    int64_t src_t = t - (w - 1 - k);
    scalar_t cx = value_at(a, base, src_t, n_bars) - mean_x;
    scalar_t cy = value_at(b, base, src_t, n_bars) - mean_y;
    cov += cx * cy;
    var_x += cx * cx;
    var_y += cy * cy;
  }
  cov *= inv_w;
  scalar_t out_v = cov;
  if (op_id == OP_TS_CORR_10) {
    var_x *= inv_w;
    var_y *= inv_w;
    scalar_t sx = sqrtf(var_x > static_cast<scalar_t>(0) ? var_x : static_cast<scalar_t>(0));
    scalar_t sy = sqrtf(var_y > static_cast<scalar_t>(0) ? var_y : static_cast<scalar_t>(0));
    out_v = cov / (sx * sy + static_cast<scalar_t>(1e-6));
    if (out_v < static_cast<scalar_t>(-1.0)) {
      out_v = static_cast<scalar_t>(-1.0);
    } else if (out_v > static_cast<scalar_t>(1.0)) {
      out_v = static_cast<scalar_t>(1.0);
    }
  }
  out[i] = sanitize(out_v);
}

template <typename scalar_t>
__global__ void cross_sectional1_kernel(
    const scalar_t* __restrict__ a,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t n_symbols,
    int64_t n_bars,
    int64_t op_id) {
  int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t columns = bsz * n_bars;
  if (col >= columns) {
    return;
  }
  int64_t b = col / n_bars;
  int64_t t = col % n_bars;
  int64_t base = b * n_symbols * n_bars + t;

  if (n_symbols == 1) {
    scalar_t v = a[base];
    scalar_t fallback = op_id == OP_CS_SCALE ? static_cast<scalar_t>(0.5) : static_cast<scalar_t>(0.0);
    out[base] = isfinite(static_cast<double>(v)) ? v : fallback;
    return;
  }

  if (op_id == OP_CS_NEUTRALIZE) {
    scalar_t sum = static_cast<scalar_t>(0);
    for (int64_t n = 0; n < n_symbols; ++n) {
      sum += a[base + n * n_bars];
    }
    scalar_t mean = sum / static_cast<scalar_t>(n_symbols);
    for (int64_t n = 0; n < n_symbols; ++n) {
      out[base + n * n_bars] = sanitize(a[base + n * n_bars] - mean);
    }
  } else if (op_id == OP_CS_SCALE) {
    scalar_t mn = a[base];
    scalar_t mx = a[base];
    for (int64_t n = 1; n < n_symbols; ++n) {
      scalar_t v = a[base + n * n_bars];
      mn = v < mn ? v : mn;
      mx = v > mx ? v : mx;
    }
    scalar_t span = mx - mn;
    bool zero_span = fabs(static_cast<double>(span)) < 1e-9;
    for (int64_t n = 0; n < n_symbols; ++n) {
      scalar_t v = zero_span ? static_cast<scalar_t>(0.5) : (a[base + n * n_bars] - mn) / span;
      if (!isfinite(static_cast<double>(v))) {
        v = static_cast<scalar_t>(0.5);
      }
      out[base + n * n_bars] = v;
    }
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
  out[i] = apply_branch_op(av, bv, cv, op_id);
}

template <typename scalar_t>
__global__ void fused_binary_branch_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t binary_op_id,
    int64_t branch_op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  scalar_t binary_v = apply_binary_op(a[i], b[i], binary_op_id);
  out[i] = apply_branch_op(c[i], d[i], binary_v, branch_op_id);
}

template <typename scalar_t>
__global__ void fused_unary_binary_branch_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ out,
    int64_t n,
    int64_t unary_op_id,
    int64_t binary_op_id,
    int64_t branch_op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  scalar_t unary_v = apply_unary_op(a[i], unary_op_id);
  scalar_t binary_v = apply_binary_op(b[i], unary_v, binary_op_id);
  out[i] = apply_branch_op(c[i], d[i], binary_v, branch_op_id);
}

template <typename scalar_t>
__global__ void fused_unary_unary_branch_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t n_symbols,
    int64_t n_bars,
    int64_t first_unary_op_id,
    int64_t second_unary_op_id,
    int64_t branch_op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = bsz * n_symbols * n_bars;
  if (i >= total) {
    return;
  }
  int64_t t = i % n_bars;
  scalar_t unary_v = apply_second_unary_window_op(a, i, t, n_bars, first_unary_op_id, second_unary_op_id);
  out[i] = apply_branch_op(b[i], c[i], unary_v, branch_op_id);
}

template <typename scalar_t>
__global__ void fused_rolling_binary_branch_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ d,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t n_symbols,
    int64_t n_bars,
    int64_t rolling_op_id,
    int64_t binary_op_id,
    int64_t branch_op_id) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = bsz * n_symbols * n_bars;
  if (i >= total) {
    return;
  }
  int64_t t = i % n_bars;
  scalar_t rolling_v = apply_rolling_max_op(a, i, t, n_bars, rolling_op_id);
  scalar_t binary_v = apply_binary_op(b[i], rolling_v, binary_op_id);
  out[i] = apply_branch_op(c[i], d[i], binary_v, branch_op_id);
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

at::Tensor fused_binary_branch_cuda(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor d,
    int64_t binary_op_id,
    int64_t branch_op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  fused_binary_branch_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      d.data_ptr<float>(),
      out.data_ptr<float>(),
      n,
      binary_op_id,
      branch_op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor fused_unary_binary_branch_cuda(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor d,
    int64_t unary_op_id,
    int64_t binary_op_id,
    int64_t branch_op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t n = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((n + threads - 1) / threads);
  fused_unary_binary_branch_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      d.data_ptr<float>(),
      out.data_ptr<float>(),
      n,
      unary_op_id,
      binary_op_id,
      branch_op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor fused_unary_unary_branch_cuda(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    int64_t first_unary_op_id,
    int64_t second_unary_op_id,
    int64_t branch_op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  fused_unary_unary_branch_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      first_unary_op_id,
      second_unary_op_id,
      branch_op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor fused_rolling_binary_branch_cuda(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    at::Tensor d,
    int64_t rolling_op_id,
    int64_t binary_op_id,
    int64_t branch_op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  fused_rolling_binary_branch_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      d.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      rolling_op_id,
      binary_op_id,
      branch_op_id);
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

at::Tensor fused_shift_unary_cuda(at::Tensor a, int64_t shift_op_id, int64_t unary_op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  fused_shift_unary_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      shift_op_id,
      unary_op_id);
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

at::Tensor rolling2_cuda(at::Tensor a, at::Tensor b, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t total = a.numel();
  constexpr int threads = 256;
  int blocks = static_cast<int>((total + threads - 1) / threads);
  rolling2_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor cross_sectional1_cuda(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.scalar_type() == at::kFloat, "native CUDA kernels currently support float32 only");
  auto out = at::empty_like(a);
  int64_t bsz = a.size(0);
  int64_t n_symbols = a.size(1);
  int64_t n_bars = a.size(2);
  int64_t columns = bsz * n_bars;
  constexpr int threads = 128;
  int blocks = static_cast<int>((columns + threads - 1) / threads);
  cross_sectional1_kernel<float><<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      a.data_ptr<float>(),
      out.data_ptr<float>(),
      bsz,
      n_symbols,
      n_bars,
      op_id);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
