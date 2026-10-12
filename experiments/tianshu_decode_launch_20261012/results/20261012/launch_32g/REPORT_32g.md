# BI-V150 32G：Decode启动配置与短边界检查

原始控制编译metadata为4 warps、3 stages；14候选×8长上下文场景，保留原attention kernel body和grid。实际交错paged KV、random physical pages、非连续Query/KV、全FP32与选定CPU、output/scratch tail及Graph计时。以下选定配置保持原TILE16，只改4warps/stage1。

|B|KV最大|baseline/ms|stage1/ms|时间变化|逐位同原生|
|---:|---:|---:|---:|---:|---|
|1|4096|0.039413|0.034518|-12.418757%|True|
|8|4096|0.215134|0.185102|-13.959908%|True|
|16|4096|0.397731|0.374521|-5.835443%|True|
|32|4096|0.781721|0.671265|-14.129900%|True|
|64|4096|1.539620|1.189508|-22.740181%|True|
|64|5120|1.917817|1.488731|-22.373646%|True|
|16|16384|1.539576|1.472631|-4.348248%|True|
|2|32768|0.728149|0.556100|-23.628231%|True|

KV32/w4/s1在长KV4K/5K微测曾快约31%，但补充14短KV/填充场景全部数值失败，不能采用。源码确认kernel和reduce_segments都依赖TILE_SIZE，只改第一kernel并不能保持完整launch契约；此处未声称该观察已完整解释每个失败元素。

保持原KV16的stage1和stage2均通过14短边界（KV1/7/13/15/16/17/31/32/33/63/64/65及B64含0-query填充）全FP32、CPU、native与tail。选stage1在B64/4K与5K微测分别约−22.74%/−22.37%，长场景逐位结果见表。

当前selected TILE16/w4/s1通过22场景数值；端到端NativeL0Prefill+Hybrid+adaptive/budget2048、C64N1284K1K候选在跑，无正式/质量新结论。Micro的时间下降不是总吞吐分数。新进程hook仅3D unquant/Q16-KV2/head128/B<=64，原Prefill路径和其zero-init hook保留。

三个完整原始包已本地分块/整包SHA256核验，清单随报告。长微测也含16K/32K少量算子场景，并未运行优化版完整16K benchmark。正式最好仍2712.55、Math102/105；性能剖析TODO。
