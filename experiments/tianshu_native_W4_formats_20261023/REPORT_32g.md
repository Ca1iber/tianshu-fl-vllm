# 非恒定原生W4 GPU格式：两种正确布局已确认，性能待测

冻结b3c02b3，8个预声明物理格式各独立GPU进程首次执行，全部进程正常退出。共同逻辑矩阵M64 K256 N256，q、K分组scale/zero及X均非恒定；每格式16384个输出用完整K INT64投影和独立BF16舍入复核，原生CPU通用INT4另用自己的转换器逐格交叉确认。代码、库、头文件、全部逻辑/物理输入的SHA前后均通过，输入存储/版本/guard保持。

精确通过的格式：generic_high_even（偶数K位于uint8高nibble）；awq_permuted_zeros_packed（N向每8列按[0,2,4,6,1,3,5,7]打包weight和整数zero，scale为BF16[groups,N]）。两者各16384个BF16输出全部一致。其余6种格式也完整执行并保留原输出，出现12288—16384个不一致，未改预期或重跑凑过。这些是物理格式假设不匹配，不能据此推断正确格式不稳定。

本结果只确认该非恒定小矩阵的实际GPU接口与布局，不是模型质量或速度成绩。下一步使用原checkpoint中的真实Down权重，对比现有BF16路径，先独立验证量化矩阵数值，再测完整调用的eager/图重放延迟；后端推荐的dotsaddz快速入口也需单独验证。

全部原始数据`/workspace/benchmark-results/20261023_native_W4_formats_GPU_tianshu`只在容器。FUNCTIONAL_GATE保存每个格式的原始结果SHA、错误/差异计数与通过集合。未改checkpoint、未接入模型、未计时或运行benchmark。
