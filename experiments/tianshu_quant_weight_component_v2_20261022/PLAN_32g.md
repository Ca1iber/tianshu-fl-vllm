# W4A16/W8A16 组件筛选

2026-10-22，沿用每日北京时间 06:00–24:00 授权，GPU 硬截止 23:58。单次组件任务最多 30 分钟。

范围：真实 checkpoint 的第 0 层 Down、LM head，M8/M64；另有 M17/N131/K259 非整齐尺寸与 strided A。W4 对称/非对称 K128 分组、W8 每通道，只压缩权重，保持 BF16 activation。使用已有只读权重格式和 CoreX 解包+BF16 dot 原型。没有原生 INT4 运算假设。

对照：同一 CUDA Graph 下 FlagGems BF16 mm_out、已有 W8A8 动态激活量化+NT 整数 dot；后者包含激活量化开销。W4/W8A16 包含内核内权重解包，不包含一次性 CPU 打包。组件 BF16 不宣称等同所有引擎分支。每 M 三组预先固定 tile，无运行时 autotune。

正确性：实际 GPU 全量解包与 CPU BF16 位级一致；同一图三次改变 activation、量化码、scale，非对称同时改变 zero point；检查全部 GPU output/FP32 reference、guards、输入/RNG不变。每次保存完整 FP32/BF16 输出及 selected W8A8 整数结果，独立 CPU 审核从原始 checkpoint 重新量化抽样列、FP64点积及尾部全输出，重新计算所有时间中位数。记录实际编译 binary/hash/寄存器/共享内存。

计时：7 组交替顺序，每组 20 次 graph replay；不包含诊断 FP32 store。先看真实形状相对当前已采用路径的收益，不以压缩率推算吞吐。只有有希望的角色/shape 才进一步集成、内存计账、跑 MATH；本轮不替换正式模型/服务。

文件原地冻结后再运行；失败保留原目录和 source，修复必须新增版本。原始权重只读。数值、性能均须独立审核通过才能引用。原始 artifacts 仅容器，代码/报告/关键结果推天数 GitHub。
