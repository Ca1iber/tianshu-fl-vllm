# 拟合 O64 G128 affine W4 GPU 组件资格

当前模型Math运行时仅CPU准备，GPU空闲后才启动。实际42层O[2048,2048]、M64、3独立输入相位；实际r4 BF16 vendor mm、已有minmax affine W4、权重拟合affine W4三路对照。拟合payload仅来自已独立验证的CPU权重研究；没有Math标签或激活校准。原checkpoint/模型/服务源不改。

GPU全42银行输入/权重/版本/边界/RNG、eager重复及poison Graph一致；层0/20/41三个相位64行全输出与独立FP64参考比较。每路七轮交替十次，排除首轮，Graph为主指标。已有原探针与验证器复用，模式名明确改为minmax/fitted。通过组件门槛需全数值/确定性合格、Graph稳态spread<1%、相对BF16至少5%降低耗时；推进拟合版本还要求相对minmax耗时不增加超过1%且随机投影误差更低。组件不是模型质量证据；模型需新鲜C64/C8 Math105>95%与4K端到端验证。GPU截止2026-10-24T15:58:00Z。
