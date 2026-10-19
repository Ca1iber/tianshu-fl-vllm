# SPDX-License-Identifier: Apache-2.0
SHAPES=(('qkv',2560,2048),('o',2048,2048),('gateup',12288,2048),('down',2048,6144))
MODES=('flag_gems_BF16','owned_BF16','BF16_storage_FP16_dot','FP16_weights_FP16_dot')
def configurations(m,n):
    if m==2048:return [dict(bm=256,bn=256,bk=32,splits=1,warps=16,stages=2,group_m=8 if n==12288 else 1)]
    assert m==64
    return [dict(bm=bm,bn=128,bk=32,splits=splits,warps=4,stages=2,group_m=1) for bm,splits in ((64,1),(64,4),(32,4))]
