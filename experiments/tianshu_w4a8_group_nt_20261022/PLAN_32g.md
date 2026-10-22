# W4A8分组NT：使用INT8 dot的新候选

W4A16原型及按原生tile补齐的W8A16均未超过Down BF16。新方向保持W4压缩存储，解包成INT8直接走已验证的INT8 dot；activation使用现有每token动态A8量化，量化在候选Graph内计时。每K128组INT32 partial乘FP32 weight scale，累加FP32后乘activation scale。不是原生INT4 dot。

只有真实第0层Down/M64及M17/N131/K259 strided/tail，两种W4格式，各12个预定tile，含2/4/8/16warp和1/2stage。候选仅组件，不替换模型。A8引入新的近似，旧Math不借用。

每个tile在同一Graph三相位改变A、packedW、scale/zero；完整format decode位级、完整GPU FP32参考、输出BF16 cast、guards/RNG/输入不变。诊断版另保存每个K块的selected实际INT32 partial；独立CPU从原checkpoint重算所有选中整数点积必须完全相同，再审核FP64缩放参考。非诊断计时路径不写partial/FP32诊断。原W8A8控制不变，每组7轮轮换×20 replay。

先离线CoreX编译，源冻结后实际GPU空闲再执行。组件提升超过10%才进一步检查全部42层/框架显存和fresh MATH，原checkpoint只读不训练。单任务30分钟，本日23:58截止，日授权06:00–24:00。代码/报告/关键结果GitHub，raw只在容器，失败不覆盖。
