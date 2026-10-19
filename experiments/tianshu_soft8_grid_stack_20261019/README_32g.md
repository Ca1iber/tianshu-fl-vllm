# 软件8位KV叠加TopP grid2（待真实服务筛选）

基准固定为已验证4K4259.87、Math102/105的软件8位KV。control和candidate均使用同一UInt8 KV/writer/arena/decode策略与原服务命令，唯一开关为TopP grid2。它此前独立BF16服务有约0.807%筛选收益，尚不能断言在新KV组合中仍有收益。

组合只执行一遍qualified base hook，复制已测grid loader仅改组合路径，按真实import链在sampler别名绑定之前安装；保留真实mask函数/原采样RNG。沿用TopP V4逐行语义门槛，未放宽：retained不变且（与现baseline完全一致或严格CPU 1e-6）。此前156 mask/8640 rows、21 RNG、8 fallback与6跨batch生命周期原件绑定；组合仍需新服务观察，若采用仍需新Math，不能复用102/105。

3+3小规模4K/C64/N64，首轮排除，逐轮不同输入seed且两模式配对一致、保存响应正文/用量/时延/前后metrics/实际42环境和源码快照。这里只是筛选，不作正式分数。新目录不改任何之前已测源。
