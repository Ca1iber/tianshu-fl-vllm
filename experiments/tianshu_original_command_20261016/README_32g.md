# 天数：按原始官方serve命令验证既有算子优化

服务命令逐项等于用户提供及公开赛题命令，不加调度预算、maxseq、stream interval、threshold、量化或投机参数。保留CV/KV16/w1/S8、native Prefill L0、75点预算内padding/gather、原生TopP和原采样。35项环境来自实际已测CV控制profile，仅记录目录重映射；未修改任何scheduler配置值。

新安装版本原始命令预计2048tokens/256seq，必须用真实Worker记录确认；扩展helper会按实际预算过滤，2048预算仍75预热点。CPU对2048个ceil映射及实际configure验证均保持预算2048/maxseq256不变；真实12hook组合未导入Torch执行通过，三个stage真实CLI解析通过。

`run_validation.py --output ROOT --reference-root /workspace/benchmark-results/20261016_combined_screen_service_tianshu --stage screen` 用未改官方benchmark脚本运行4K/1K/C64/N64四轮排首轮；只是快速工程小测，不是正式N256成绩。

screen完整通过后，`--stage math --screen-root SCREEN`按官方Level3/C8/105题评测；新Math严格>95%后，`--stage official --screen-root SCREEN --math-root MATH`原脚本N256四轮排首轮。每步均重新启动原始serve命令、验证源码/安装源/实际Worker/75预热/算子真实路径/零preempt、保存全部原始产物。不借旧Math或自选参数成绩。完整比赛资格还需最终16K与复现等要求，不用单一4K证明全部资格。

2026-10-16已通过CPU及实际原命令N64运行，44源/75warm/真实2048和256seq均本地审计。N64重复seed=0使后三轮大量命中prefix cache，5510.08不代表正式N256，不采用为新最佳。详见results/20261016/screen64_32g；独立原命令Math105已启动。
