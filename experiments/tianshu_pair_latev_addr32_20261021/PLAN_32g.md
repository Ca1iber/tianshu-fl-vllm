# 成对K后的地址32与延后V加载：四路因子对照

基于当前f1df731b、真实K8V16 12288B、T16/w1/s1、原分段/.cv。四路current、addr32、late_v、late_v_addr32；只在3D、n25..64、QBF16/Kuint8/VBF16、KV_NONE、无TD/滑窗/特殊mask的已声明范围启用。模型dtype/权重/KV数据与所有QK/softmax/PV数学表达式保持，预期所有保存的完整GPU输出必须逐位相同；未选范围使用原wide kernel。

late_V将V offset/load/cast移到softmax/alpha后、PV前。精确可逆文本替换恢复父源码字节；LATE=false AST完全等于父源，函数签名不改。它可能减小活跃区间，也可能损失prefetch，必须测耗时而非按regs推断。当前reader完整副本qualified_current_reader.py与原f1df SHA相同，属于代码，便于GitHub复现，不拷贝权重。

addr32只改整数注解和cast，独立AST归一化证明无浮点/物理pointer/slot_mapping变更。旧7b94 S16上的kv32/all32曾−.086%/−1.026%，未采用；本轮是配对K后的f1df，保留旧失败。每次launch用原signed元素offset guard（2048×sumstride masked余量）选窄版；producer和reducer必须共享选择结果，宽shape/stride/非支持范围都使用原64版本。CPU验证真实选择器/返回/grid/参数透传与宽fallback，不分配>2Gi-element GPU缓存，不冒称该规模GPU覆盖。

14场景、两阶段同Graph变数据，四路预计112完整GPU/304独立CPU向量/1216项原CPU检查，九次正反交替×20热Graph，504组计时，含真实writer+attention+reducer。含40/16K单层和length1捕获至4K/8193，仅组件而非完整16K。原.002+.01误差阈值保持，另严格全输出bitwise。保留全部原始慢路/编译与数值失败。CPU门禁先绑定407源、父f1df和B64S4，再GPU冻结，真修复新目录。

只有C64/4K和5K明确同次收益，才考虑现行407源的独立服务N64×3；新profile需fresh Math>95%，C8评测不冒称覆盖n>24候选，C64另有全GPU/CPU及真实Graph证据，之后才N256正式。代码/报告/关键JSONCSV仅容器+GitHub，原PT/日志仅容器，不改用户dirty、FlagGems、沐曦，无子agent。23:58北京GPU硬停。

CPU初测已通过源码与2816正向路径，在宽回退fixture缺NUM_SEGMENTS_PER_SEQ必需参数时退出；这是CPU用例不完整，未启动GPU。原日志保留，补齐参数和实际T16w1s1范围，并显式拒绝其它tile/warp/stage与非每seq一个query。没有改数值/bitwise阈值或原宽回退实现。
