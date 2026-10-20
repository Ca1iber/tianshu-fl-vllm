# Paired-K读取：4K N64同配置服务对照

沿用正式future-KV准入版的52项环境、261份基础来源、原始serve argv、4096预算、83点预热、128MiB暂存区、max_num_seqs64、async调度、K8/V BF16和长上下文S16。唯一开关是TIANSHU_PAIR_LOAD_MODE，candidate仅把实际3D decode producer内层kernel切到GPU逐位通过的pair_rows源；2D/prefill、writer、reducer、采样与RNG保持。

两个模式均通过相同外层路由，配置和冻结来源必须一致。旧S16绑定记录发生在此新外层之前，仅证明内层初始化；新binding记录实际使用的kernel SHA、Graph键与捕获。审计把这些捕获映射到真实B64、computed>=4096的模型Graph重放，禁止将Python捕获次数当设备重放次数。

每模式3轮4K/1K C64 N64，事前排除首轮，输入seed各轮不同且两模式相同。两次seed777 canary只校验单模式可重复，跨模式文本相同仅诊断，不是额外质量门槛。完整192+192请求、原argv/环境/源码、KV/RNG/原生采样、零preemptions以及实际新kernel绑定均需独立复核。原始回复留容器，GitHub只留代码/报告/关键结果。GPU组件6.75%不是模型成绩，尚无新Math或正式分数。
