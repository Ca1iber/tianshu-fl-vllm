# 软件8位KV：固定decode策略与有界prefill缓存

独立runtime，前置decode七配置/十场景原始结果与109来源冻结。保持同一软件8位format、实际writer、BF16 Q/output和FP32 attention算法；固定S策略在dispatch_policy.py，始终T16/w1（B<=64）。B>64保留旧byte实现。

prefill临时解码到单一32768页BF16 arena，512MiB加一guard页、128KiB页表。一次分配、不同形状同指针复用；超过逻辑页数上限保留原byte路径，mixed先判断容量，避免重复处理decode。完整engine接入将把reserve放在实际profile_run内，使可用KV内存计量包含该缓冲区；目前组件尚不证明engine集成。

CPU检查1280个policy边界、7容量边界、4实际mixed dispatch路径（包括容量fallback仅调用一次byte decode）、67242个dense页地址、同一编码穷举、spec/allocator相关门槛。GPU计划18场景：原16加B1/4K与实际强制容量fallback，包含真实metadata/writer/forward、同Graph变输入、完整writer和有效dense解码bitwise、输出/segment尾部与RNG、FP32/CPU head参考，7×20回放包含全部写入/解码/attention。只降低组件退化不等于模型准确率或正式分数。
