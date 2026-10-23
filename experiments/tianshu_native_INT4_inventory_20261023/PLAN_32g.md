# 天数PyTorch原生INT4接口：先功能，再性能

实际安装torch2.10的CUDA调度注册包含_convert_weight_to_int4pack、_weight_int4pack_mm，以及额外的AWQ/dotsaddz入口。CPU注册查询及_meta_registrations源码SHA已保存，查询未初始化CUDA。注册不等于BI-V150可运行，更不代表有吞吐收益。

第一步只测通用packed入口的4个确定性64×256×256小矩阵：BF16 A恒为1/8，4bit unsigned q取7/8/9/10，scale=1/8、zero=0、group128，按signed中心8的假设预声明所有输出分别为-4/0/4/8。两nibble相同排除高低打包顺序歧义，数值全为二进制可精确表示。若该入口语义或支持不同，原样记录失败，不修改预期凑过。不会据此宣称真实权重量化准确或性能合格；通过后才另行准备真实模型weight/完整独立参考与计时。

首次GPU前冻结脚本、CPU元形状/注册/预期；等待当前v3模型质量与短测结束，GPU空闲才单独执行，90秒上限、15:58 UTC硬停止前至少5分钟。不动checkpoint、原r3文件或运行环境；本轮不改模型服务，不进行Math/benchmark。
