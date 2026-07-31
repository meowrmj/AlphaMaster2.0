#include <torch/extension.h>

torch::Tensor elementwise2_cuda(torch::Tensor a, torch::Tensor b, int64_t op_id);
torch::Tensor elementwise3_cuda(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t op_id);
torch::Tensor elementwise1_cuda(torch::Tensor a, int64_t op_id);
torch::Tensor shift1_cuda(torch::Tensor a, int64_t op_id);
torch::Tensor rolling1_cuda(torch::Tensor a, int64_t op_id);

torch::Tensor elementwise1(torch::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  return elementwise1_cuda(a.contiguous(), op_id);
}

torch::Tensor elementwise2(torch::Tensor a, torch::Tensor b, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return elementwise2_cuda(a.contiguous(), b.contiguous(), op_id);
}

torch::Tensor elementwise3(torch::Tensor a, torch::Tensor b, torch::Tensor c, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "a, b and c must have the same shape");
  return elementwise3_cuda(a.contiguous(), b.contiguous(), c.contiguous(), op_id);
}

torch::Tensor shift1(torch::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3, "a must be [B,N,T]");
  return shift1_cuda(a.contiguous(), op_id);
}

torch::Tensor rolling1(torch::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3, "a must be [B,N,T]");
  return rolling1_cuda(a.contiguous(), op_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("elementwise1", &elementwise1, "AlphaMaster elementwise unary CUDA ops");
  m.def("elementwise2", &elementwise2, "AlphaMaster elementwise binary CUDA ops");
  m.def("elementwise3", &elementwise3, "AlphaMaster elementwise ternary CUDA ops");
  m.def("shift1", &shift1, "AlphaMaster shift CUDA ops");
  m.def("rolling1", &rolling1, "AlphaMaster rolling CUDA ops");
}
