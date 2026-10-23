# v1启动检查失败，未执行任何Math题目

冻结4efb8be。模型加载42个W4 bank、42个load预热、42个实际Worker profile scratch探针全部完成；新增payload280756224B，实际模型内存7669493248B，唯一参数/buffer storage7600724640B，源与pack输入校验通过。但新增W4捕获bank为0，原native_down记录中M64仍为42个BF16捕获。因此独立startup审计拒绝，evaluator_pid为空，没有发出Math请求，没有模型准确率成绩，不采用。

原因：替换UnquantizedLinearMethod.apply中的Python M==64分支发生在已经被torch.compile追踪的旧custom-op外部，实际图仍调用原tianshu_native_down::linear。不能把42个零输入预热当成模型实际路径命中。保留本版本、真实record和失败结果，不修改这些已执行的来源或事后让审计放行。

修正方向：v2保留实际图中原custom-op的schema/fake/graph target，通过CustomOpDef.register_kernel('cuda')更换该实际算子的CUDA路由，仅M64调用W4，其他M委托保留的真正原实现。核对原OPDEFS._init_fn及其banks closure与真正layer的一致性。新W4审计仍要求42个真实capture，原BF16 capture必须为0。

原始目录 `/workspace/benchmark-results/20261023_dots_Down64_math64_tianshu`；服务与质量均已停止，GPU空闲。不存在对同一准确率版本重跑追分；v1连评测都未进入。
