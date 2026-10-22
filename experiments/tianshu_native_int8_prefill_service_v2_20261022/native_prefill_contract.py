# SPDX-License-Identifier: Apache-2.0
SUPPORTED_ROWS=(512,4096)
def use_native(mode,m,k,contiguous):
    assert mode in ('control','candidate')
    return mode=='candidate' and m in SUPPORTED_ROWS and k==2048 and contiguous
