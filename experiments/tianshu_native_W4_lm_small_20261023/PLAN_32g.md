# LM Head M1..63原生W4与当前实际Embedding.apply对照

当前r3 LM64为已采用W8A8，M<64为BF16回退。本组件只测M1/2/4/8/16/32/48/63，不拿M64的BF16当现用对照。真实lm_head[130560,2048]、前版已验证dotsaddz group128 WD/S4/ZD及3phase独立合成X和完整FP64原始/反量化参考，全部逐SHA复用，不重新量化或改checkpoint。

真实小批次对照直接调用UnquantizedEmbeddingMethod.apply，在r3完整profile环境下按WorkerFL方式enable FlagGems（空名单）。预热阶段用Python frame trace核对实际vendor linear及源码SHA，保留精确调用路线。LM当前在主模型Graph外，因此eager为主要组件结论，Graph为补充；不能直接推导模型收益。

所有X及FP64参考先按行roll1再截前M，避免原fixture首行为0导致M1只测零输入；不改变参考方程。每M×3phase全部输出，BF16对原始FP64、W4对BF16反量化权重FP64逐格验证：absdiff<=.012abs(ref)+.006RMS且relativeL2<=.006，沿用旧阈值。Graph清NaN后/eager逐字节一致、有限，全部输入/权重/pack/scale/zero全字节版本guards、RNG/checkpoint/source不变。完整所有输出的相对原BF16量化误差单列；组件通过不等于新模型Math通过。

各M eager/graph各7轮排首轮，后6轮中位数，20次调用，轮次交替顺序，保存每轮结果和spread。新GPU目录不覆盖旧记录；900秒上限，15:58UTC前停止。只组件，不跑模型/官方benchmark/16K；有实质收益后另行新质量与同期短测，不借用旧模型成绩。原始数组/日志仅容器，源码/报告/关键结果推天数GitHub。
