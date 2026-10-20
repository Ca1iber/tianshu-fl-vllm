# Grouped-M prefill 组件独立 CPU 复核

沿用 row-major prefill 的真实 checkpoint 行、全部 A 量化、实际 GPU 选中 INT64/BF16、source snapshot、七轮交替计时与完整 GPU 检查记录复核，扩展8种 M×12个配置。额外按同一 tile/M 成对比较 group1 与 group8；只总结执行顺序差异的组件时间，不把缓存复用动机当成已测硬件结论。未运行 GPU/服务/Math。
