// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>

at::Tensor weak_alias(const at::Tensor& input) {
  TORCH_CHECK(input.layout() == at::kStrided, "weak_alias requires strided input");
  TORCH_CHECK(input.numel() > 0, "weak_alias requires nonempty input");
  return at::from_blob(input.data_ptr(), input.sizes(), input.strides(),
                       [](void*) {}, input.options());
}
