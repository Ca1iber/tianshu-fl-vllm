# 双 query 验证的分段数 sweep

复用已经通过14阶段 FP32/CPU 检查的 native_verify，只改变 S4/8/16/32 与足够的resident scratch容量；tile、warp、attention/reducer算术保持原样。真实C64 Q2及ragged Q1/Q2、随机分页、strided cache、Graph输入/长度变更和CPU oracle均要通过。

当前试验仅元件筛选，不替代原生ngram的整链服务、fresh Math和正式benchmark。GPU只在ngram整链pilot退出且实际空闲后运行，15:58UTC硬停。
