# SPDX-License-Identifier: Apache-2.0
"""Restrict the new precision to the captured uniform64 full-decode graph."""
def eligible(role, shape, dtype_bf16, contiguous, mode_name, num_tokens,
             num_reqs, uniform, has_lora):
    return (role in ('QKV', 'O') and tuple(shape) == (64, 2048) and dtype_bf16
            and contiguous and mode_name == 'FULL' and num_tokens == 64
            and num_reqs == 64 and uniform is True and has_lora is False)
