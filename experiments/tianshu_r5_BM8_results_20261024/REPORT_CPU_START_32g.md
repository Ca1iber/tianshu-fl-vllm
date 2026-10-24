# r5+BM8组合：CPU已通过，开始新质量评测

截至北京时间2026-10-24 16:45左右，新候选CPU_GATE完整、model_ready=true：1022个冻结来源加门禁自身共1023个运行来源，包含原正式r5全部997来源，逐文件SHA保持一致。67项显式运行环境、原始13项serve argv，两组QKVO fitted模式都保持candidate，BM8模式单独切换。旧8044544元素bitwise组件资格只用于证明相同BM8内核复用，未将旧Math或吞吐分数视作新组合成绩。

新模型代码在../tianshu_r5_BM8_KV_20261024；首次清单格式读取失败及修复见其REPORT_PREP_32g.md。候选读取M64 3D分块由BM16/Q2/grid96切为BM8/Q1/grid64，保持原K8/V BF16格式、所有权重及其它M路径。实际启动仍须严格校验最终源码、机器码、寄存器、网格、真实Graph捕获/重放；旧pair层记录不能替代最终路径记录。

启动前原r5 997来源SHA通过、384个固定编译缓存齐全（restored=0,missing=0），GPU空闲。新鲜C8→C64每组完整105题顺序启动，raw与日志均在容器/dev/shm/tianshu_r5_BM8_runs_20261024。控制器包含北京时间23:58截止及异常清理；本记录不代表实际模型已就绪、不代表新正确率已通过，不代表有新吞吐。须按后续state/result、profile_audit和accuracy_audit核验，结果在GPU停止后压缩归档。

正式已发布版本、tag与6123.31 total tok/s保持不动。框架层已验证进展见../tianshu_framework_status_20261024/REPORT_32g.md；本BM8候选属于量化KV的读取执行路径，不将其潜在收益计作调度器收益。无主机代码、数组或日志副本。
