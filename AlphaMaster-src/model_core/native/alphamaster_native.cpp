#include <torch/extension.h>

at::Tensor elementwise2_cuda(at::Tensor a, at::Tensor b, int64_t op_id);
at::Tensor elementwise3_cuda(at::Tensor a, at::Tensor b, at::Tensor c, int64_t op_id);
at::Tensor elementwise1_cuda(at::Tensor a, int64_t op_id);
at::Tensor shift1_cuda(at::Tensor a, int64_t op_id);
at::Tensor fused_shift_unary_cuda(at::Tensor a, int64_t shift_op_id, int64_t unary_op_id);
at::Tensor fused_binary_branch_cuda(at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d, int64_t binary_op_id, int64_t branch_op_id);
at::Tensor rolling1_cuda(at::Tensor a, int64_t op_id);
at::Tensor rolling2_cuda(at::Tensor a, at::Tensor b, int64_t op_id);
at::Tensor cross_sectional1_cuda(at::Tensor a, int64_t op_id);

at::Tensor elementwise1(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  return elementwise1_cuda(a.contiguous(), op_id);
}

at::Tensor elementwise2(at::Tensor a, at::Tensor b, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return elementwise2_cuda(a.contiguous(), b.contiguous(), op_id);
}

at::Tensor elementwise3(at::Tensor a, at::Tensor b, at::Tensor c, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(), "a, b and c must have the same shape");
  return elementwise3_cuda(a.contiguous(), b.contiguous(), c.contiguous(), op_id);
}

at::Tensor shift1(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3, "a must be [B,N,T]");
  return shift1_cuda(a.contiguous(), op_id);
}

at::Tensor fused_shift_unary(at::Tensor a, int64_t shift_op_id, int64_t unary_op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3, "a must be [B,N,T]");
  return fused_shift_unary_cuda(a.contiguous(), shift_op_id, unary_op_id);
}

at::Tensor fused_binary_branch(at::Tensor a, at::Tensor b, at::Tensor c, at::Tensor d, int64_t binary_op_id, int64_t branch_op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(c.is_cuda(), "c must be a CUDA tensor");
  TORCH_CHECK(d.is_cuda(), "d must be a CUDA tensor");
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes() && a.sizes() == d.sizes(), "all tensors must have the same shape");
  return fused_binary_branch_cuda(a.contiguous(), b.contiguous(), c.contiguous(), d.contiguous(), binary_op_id, branch_op_id);
}

at::Tensor rolling1(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3, "a must be [B,N,T]");
  return rolling1_cuda(a.contiguous(), op_id);
}

at::Tensor rolling2(at::Tensor a, at::Tensor b, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3 && b.dim() == 3, "a and b must be [B,N,T]");
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return rolling2_cuda(a.contiguous(), b.contiguous(), op_id);
}

at::Tensor cross_sectional1(at::Tensor a, int64_t op_id) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(a.dim() == 3, "a must be [B,N,T]");
  return cross_sectional1_cuda(a.contiguous(), op_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("elementwise1", &elementwise1, "AlphaMaster elementwise unary CUDA ops");
  m.def("elementwise2", &elementwise2, "AlphaMaster elementwise binary CUDA ops");
  m.def("elementwise3", &elementwise3, "AlphaMaster elementwise ternary CUDA ops");
  m.def("shift1", &shift1, "AlphaMaster shift CUDA ops");
  m.def("fused_shift_unary", &fused_shift_unary, "AlphaMaster fused shift+unary CUDA ops");
  m.def("fused_binary_branch", &fused_binary_branch, "AlphaMaster fused binary+branch CUDA ops");
  m.def("rolling1", &rolling1, "AlphaMaster rolling CUDA ops");
  m.def("rolling2", &rolling2, "AlphaMaster rolling binary CUDA ops");
  m.def("cross_sectional1", &cross_sectional1, "AlphaMaster cross-sectional CUDA ops");
}
