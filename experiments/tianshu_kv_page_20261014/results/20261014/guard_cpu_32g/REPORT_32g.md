# 天数 BI-V150 32g：KV page16/32 准入条件与组合钩子预验证

候选目标是框架KV页长16→32，并保留已经验证的guarded KV16/w1、原生GQA Prefill M256/KV128/w16/s1/L0、75点预热和gather等优化。当前原生backend允许MultipleOf16，统一attention把v.shape[1]传为BLOCK_SIZE；限制来自hybrid.supported固定k.shape[1:]为(16,2,128)。直接改服务参数会丢失已验证原生预填充路径，不构成有效A/B。

geometry_guard.py先检查当前hybrid完整源码SHA5c55a8d...，只将该条件扩展为page16或32，其余源码逐字保留。hooks/sitecustomize.py先加载选定的S4或S8基线，然后通过既有NativePrefillFinder返回的loader组合外层loader；不能用raw PathFinder绕过原生PrefillLoader。载入之后才替换准入函数，首次接受形状记录页长并移除记录包装，后续直接执行扩展函数。安装源码和内核体不改。

CPU独立检查69项通过：原16行为、32新增准入、其它页长回退、dtype/device/量化/窗口/sink等不支持条件仍拒绝、精确仅一处条件替换；另用CPU stub核对对既有父loader的委托、dispatch对象身份保持及一次形状记录后去掉包装。isolated -I进程还实际执行了S4和S8完整hook注册链，均保留唯一NativePrefillFinder，没有import torch或vLLM。

这些结果只证明准入及加载器组合逻辑，不证明真实GPU模块执行、32页数值、CUDA Graph或服务收益。本候选尚未运行GPU测试、尚未接入当前服务。当前B64S8正式4K优先，五份formal和三份handoff冻结源码保持不变。

上卡前仍需要用相同逻辑KV数据、打乱物理页表，对page16/32做独立FP32/CPU参考、因果Prefill/混合批次/Decode、尾部31/32/33、padding/stride和图重放比较；测4K/5K微场景，不跑完整16K。通过后才做只差--block-size16/32的独立小4K对照；明确收益后重新完整Math105严格>95，再运行原版正式4K。基线用S4还是S8应由当前正式结果决定。
