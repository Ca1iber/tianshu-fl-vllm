# SPDX-License-Identifier: Apache-2.0
"""Pure host partition size, also used by the real launch wrapper."""
def chunk_size(k,splits,bk):
    assert all(isinstance(v,int) and v>0 for v in (k,splits,bk))
    return ((k+splits*bk-1)//(splits*bk))*bk
