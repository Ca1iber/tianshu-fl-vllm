# Prefill INT8 模型接入前的实现约束（尚未实施）

row-major NN 大M组件已通过36配置/108数值阶段与独立CPU核验，相对当前BF16 linear：M128约+37.36%、M2048约+11.35%、M4096约+13.03%。量化相对L2约1.17%。grouped-M CPU证明已过，GPU仍在进行，必须先终局与独立CPU复核，不能用进行中的最佳值选作模型定稿。

可复用本轮 NN 物理 bank 安装：逻辑[N,K]保持，stride=[1,12288]，实际 contiguous 转置交给NN；43-bank总payload不增加。若prefill采用NN且decode保留原NT，须真实保留双bank并在原生DeviceMemoryProfiler内计入新增约1.06GB，不可绕过内存计量。较简单的最小配对是两组都继承同一NN decode布局/分块，只有prefill BF16→INT8变化；该NN decode不是正式推荐，必须声明对照基线。

prefill当前在opaque旧INT8 linear的cfg=None分支，交给prefill_route闭包执行BF16；只包旧kernel.mm拦不到prefill。可在已装UnquantizedLinearMethod.apply外层接入新的opaque custom_op，其实际实现内部按运行M选择prefill量化或委派旧apply。Fake返回形状仍[M,12288]，避免在Dynamo trace中按符号M分支；银行在loader process阶段注册，不在trace中修改字典。M8/M64继续旧route，LM完全不拦截。

激活顺序：先完成原生/父方案的合格BF16 warmup、capture与NN decode原捕获，再预热新prefill内核，最后开启请求路径。两组预热相同内核和临时buffer，完整核验CPU/GPU RNG不变。可使用实际capture/padding的M集合预热；M是constexpr，不能只热6个组件点然后请求中对每个稀有M现场JIT。运行M不在明确已预热集合则委派BF16，记录fallback，避免未知M热路径编译。内存释放后等待真实GPU空闲和审计终局。

审计需明确适配：当前父qualifed audit在ready实际请求中要求42层走BF16 linear_contiguous，prefill INT8候选自然不满足。不能用startup dummy run伪装请求命中，也不能删除无关源码、bank、FutureKV、attention、Graph、抢占或环境检查。应独立复制/适配那一处prefill路由判断，并用候选42层实际INT8源码、配置、二进制、M与请求阶段记录替代；保留旧warmup42层证据及所有其他父审计。旧dense binding kernel/config仍只描述原NT模块，实际NN以新增route为准。

prefill改变模型量化范围，跨组canary文本不必相同；每组重复同seed必须可复现，RNG行为/usage/有限输出必须验证。先4K/1K C64 N64三轮排除首轮，收益明确才新Math105题（严格>95%）及正式N256 4K。最终Math必须实际覆盖候选prefill路径，不沿用旧decode-only Math，不为失败配置重复抽样求过线。不做常规16K。

所有代码、报告、关键结果只放天数容器与GitHub。用户保留的两个gateup_silu文件不改不提交。预约截至2026-10-22 00:00北京时间，GPU硬截止23:58。
