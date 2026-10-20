# K8V16的32位合并读取：GPU验证通过，性能收益不足，不采用

保持12KiB页、K软件8位/V BF16值、现有长上下文S16分段和全部采样逻辑。三个独立变体分别把4个K字节、2个BF16 V元素或两者用uint32字读取再恢复原位模式；不是把V量化成int8，也不是整数QK。目标是减少地址/加载开销，是否有益必须实际测量。

CPU bytearray/struct独立检查连续/跨页stride、反向页表、131072个K字节及131072个V原始BF16位模式（包括全部65536种BF16位模式）；存储保持。三个源码变换可精确反转到当前7b94...S16源，只有kernel_unified_attention的加载块变化，函数ABI、reduce、KV解码和int64偏移保持。要求cache指针4字节对齐、K前三维stride整除4、BF16 V前三维stride为偶数、末维连续；组件入口检查这些前提。

probe.py准备4组同数据对照、12场景（主4K/5K、ragged/stride，B40/16K，B41跨8192边界），两相位同Graph改变Q/K/V/页表/长度/slot，完整GPU FP32/CPU选定向量、整个缓存字节、输出/segment保护区、RNG和相对control逐位输出检查。热Graph计时仅一层decode加一次KV更新，9次交替顺序×20 replay；没有模型或服务分数。

当前只有CPU_GATE_32g.json通过，尚未执行任何GPU kernel或速度测量；须等当前正式8轮和后续KV水位对照释放GPU。原始首次CPU代码/数据留20261020_packed_load_CPU_tianshu，带可重运行输出参数的入口及最新CPU检查留20261020_packed_load_CPU_v2_tianshu。没有改动当前225份Math/正式冻结源或234份排队水位源。

14:21 GPU结果更新：上述CPU准备状态为执行前记录。12场景×4组×2相位完整GPU参考及全部CPU选定向量、缓存字节、RNG/输出/segment保护区、相对control逐位输出均通过。B64/4K控制0.589570ms、K32 1.004294ms（吞吐-41.30%）、V32 0.587221ms（+0.3999%）、KV32 1.002659ms（-41.20%）；B40/16K V32仅+0.195%。未接入服务、未跑新Math、不采用，当前正式版本保持。详见GPU_GATE_32g.json、COMPONENT_ROUNDS_32g.csv和可重运行的verify_component.py；原始event/张量/快照留容器20261020_packed_load_decode_tianshu。
