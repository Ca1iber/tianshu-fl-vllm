# BI-V150 32G：GateUp/SiLU首版数值失败

8场景×7配置共56条结果，14条小inter130场景通过，完整inter6144/K2048模型形状全部数值失败。无compile/runtime错误，output尾部保留。部分完整shape计时−26.8%但数值错误，禁止作为有效收益或送入真实模型。

保留真实compiled epilogue的BF16 projection→FP32 SiLU/product→BF16 store舍入边界、独立GPU FP32与选定CPU/native对照。失败根因尚未证实；不凭布局/编译猜测写结论。当前不采用，无新服务/Math/正式测试。

原始56配置数值、时延样本、编译epilogue、代码和source bundle完整1867300 bytes，SHA256 d54a2c7af6a4d81aaa645eb324ea3d2d9319e00c6f7c6a15169b244c2ebb2ac4分块及整包本地核验。正式最好不变。性能剖析TODO；未改安装源码/权重/原始benchmark或沐曦仓库。
