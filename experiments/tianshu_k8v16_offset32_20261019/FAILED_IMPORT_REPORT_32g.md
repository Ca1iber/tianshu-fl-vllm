# 地址32 V1：导入失败，零数值和计时场景

2026-10-19，来源c6bbc4c。组件配置了Python路径后，`from source import load_variant`
错误地解析到另一个实验目录tianshu_soft8_engine_v2_20261019/source.py，触发ImportError。
失败发生于数值数据及timing初始化之前，没有有效性能结论，也没有采用候选。
原代码和日志完整保留；后续独立V2明确按文件路径加载私有helper，并增加真实导入语句的CPU检查。

失败原件 `/workspace/benchmark-results/20261019_k8v16_offset32_component_tianshu`。
4921075字节、19分片及整包SHA已本地核验：
`e4b79de741b66aa607b67b522d05bb955fb7939510feac45d0bdf25a870d3167`。
本地k8v16_offset32_failed目录的local_failure_audit.json确认failed、无micro/result、
明确ImportError和GPU上下文退出。未修改已验证4997.05 profile。
