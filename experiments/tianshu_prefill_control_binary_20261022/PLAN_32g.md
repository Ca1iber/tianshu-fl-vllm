# 预填充控制的原始binary来源复现

新测M128因为完整cubin匹配失败停止。旧资格M128cache来自large_tiles/kernel.py，新默认重编译来自large_tiles_v2/kernel.py；cache key相同且ELF全部ALLOC sections一致，debug路径不同。三个M的旧缓存并非全都保留资格cubin，因此不覆盖已有cache，不放宽门槛，而在独立cache尝试原V1/V2两份设备函数，CPU AST必须相同。

仅零输入GEMM支持性/编译诊断：M128/512/4096，正式逐M配置，Q/W全0、scale1，unused Integer/Projected均实际2D，SME资格同原探针。每版本独立cache，记录完整cubin/metadata/代码hash、输出0及RNG，不能当成数值或性能评测。选择每M与正式gate完整cubin SHA相同的cache；若未匹配则明确失败，不能继续测错控制。新缓存各文件hash冻结，后续单独完整比较仍有三变化输入和独立CPU。GPU前空闲，最长300秒，今日23:58停，原缓存/源码不修改。代码与关键证据GitHub，cache只容器。
