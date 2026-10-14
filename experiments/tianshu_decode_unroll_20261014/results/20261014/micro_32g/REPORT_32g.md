# 天数BI-V150 32g：KV循环展开对B64明显变慢

固定已采用.cv/S8/guarded KV16/w1，原.cv对照，新副本只添加KV_LOOP_UNROLL constexpr和唯一for循环头的tl.range unroll hint，整数宽度、循环算术体、reducer、tile/warp/stage/grid不变；生成源可逐字逆还原8deb。17场景×4候选共68例数值与变QKV Graph、CPU参考及尾部检查通过，765原始event和源/编译产物全部下载复核。因子1身份控制逐位一致；其它配置部分不逐位，不冒充精确。

|batch|KV最大长度|factor1|factor2|factor4|factor8|
|---:|---:|---:|---:|---:|---:|
|8|4096|-0.1978%|+6.7248%|+7.4372%|+6.8596%|
|16|4096|-0.0873%|+6.9337%|+7.6665%|+8.5118%|
|24|4096|-0.1819%|+5.2886%|-5.7203%|-5.1611%|
|32|4096|-0.0447%|+8.2497%|+9.2788%|+11.6760%|
|64|4096|+0.0135%|+10.1047%|+13.8902%|+21.8647%|
|64|5120|-0.0447%|+10.0802%|+13.4023%|+21.2646%|
|64|65|+0.0118%|-0.0320%|-0.9152%|-0.8453%|

B64/4K因子2/4/8慢10.10%/13.89%/21.86%，5K慢10.08%/13.40%/21.26%，不采用、不进入服务/Math/正式。B24因子4局部减时5.72%不能外推C64吞吐，保留记录。

核对实际B64编译LLVM，原版/factor1/factor2/factor4/factor8静态exp2 call分别8/8/24/40/72，LLVM文本约112/112/241/353/577KB，BF16 load intrinsic也随之增加；证明生成代码确有扩展，不只是记录传入hint。真实nregs85/85/115/130/162，均0spill。静态调用数不是实际执行指令数，也不由寄存器数推断occupancy或硬件根因。详见ACTUAL_LLVM_CODE_EXPANSION_32g.json。

原始归档4022393bytes/16parts SHA256 `588e12d7d6756e577f032f2c2950989193c5a2aec275f623d9dd3fe3cf37608c`。原件service-results/20261014_tianshu/decode_unroll/20261014_decode_unroll_tianshu/micro，verify_downloaded.py --root <raw>/micro验证原始计时/参数/源；后验LLVM核对另保存。当前正式最佳4091.29与Math103保持。

下一独立方向为已采用native TopP的block/trunc tile及warp32：已读实际wrapper，scratch buffer按vocab而非tile分配，kernel BLOCK_SIZE/BLOCK_SIZE_TRUNC为constexpr，可单独覆盖；原微测只扫w2/4/8/16而未扫tile或w32。需以已采用8192/4096/w16/s1为对照，严格保留p=.95/k和RNG，做相对掩码及独立CPU语义检查；已有baseline部分CPU严格误差必须保留，不能放宽1e-6或把不同掩码强行视为等价。当前尚未实现/GPU执行该新方向。
