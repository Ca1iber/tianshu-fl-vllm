# Down W8A8：42-bank MLP复测通过，下一步模型质量

使用全部42层真实原权重，GateUp完全沿用r1实际路径：M64原融合、M128原范围融合、M512/4096原生INT8；各M均核对完整cubin。候选只把Down改为原生NN W8A8、单一新NN bank，原BF16 Down保留。三阶段共享合成activation分别送42个独立真实MLP，属于42-bank组件，不是完整Transformer运行。

|M|原完整MLP ms/layer|候选 ms/layer|组件加速|
|---|---:|---:|---:|
|64|.135338|.124681|+8.547%|
|128|.160364|.157045|+2.114%|
|512|.380743|.336097|+13.284%|
|4096|2.564902|2.129130|+20.467%|

独立Down-only42-bank计时也保存，使用同一共享隐藏输入隔离42份权重工作集，不能混同逐银行真实GateUp隐藏分布的whole-MLP计时。全部结果见KEY_TIMINGS_32g.csv。两臂7组交错，每组3次whole42图，表格按42归一化。

GPU对504个bank/phase，候选完整INT32及量化BF16与独立原Triton diagnostic逐位一致；两臂whole42图污染所有输出和隐藏区后重放，完整输出哈希=eager，保护区/输入/RNG不变。42份GU/Down/新INT8 bank前后哈希一致，原checkpoint前后相同。独立CPU重算全部42份Down W8，504份完整选中K6144行的A8/scale，以及129,024个INT64/量化BF16和原BF16-FP64参考。

相对原BF16的归一化L2误差约2.64%..5.57%，逐层指标保留。它不是模型正确率，不能复用r1的101/105。尚无本候选模型/Math/服务吞吐/官方4K/16K成绩。下一步先新模型质量；候选未采用，5777.22 tok/s正式版冻结不变。

raw仅容器 /workspace/benchmark-results/20261022_native_int8_down42_GPU_tianshu；源码、计划、关键CSV和独立CPU gate保存天数GitHub。原24点cache仅复制到独立目录，原cache字节未修改。
