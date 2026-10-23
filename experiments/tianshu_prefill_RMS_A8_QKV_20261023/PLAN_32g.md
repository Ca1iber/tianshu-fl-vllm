# prefill RMS+A8与原生INT8 QKV完整链：CPU输入准备阶段

本目录首先在CPU只读原checkpoint，为真实layers0/20/41拼接原q/k/v权重到N2560×K2048，独立重算逐行对称INT8权重及FP32 scale，逐字节匹配当前已合格r3实际bank的original_BF16/packed/scales SHA。只保存三份各4096字节的真实input norm权重到容器CPU结果目录，不复制完整模型/权重，不导入Torch/Triton/vLLM/FlagGems。此CPU准备不表示组件已实现或已通过GPU。

下一步另写并冻结GPU probe及独立CPU verifier后才可运行。固定复用原IR RMS+A8 kernel（4warps/1stage，K2048 eps1e-6，不改数值），对照为安装版IR-native impl_fn的独立torch.compile(norm)→原A8→原Torch CoreX _int_mm→原_dequant，候选仅把norm与A8融合，其余原NN pack、INT32 native mm和dequant顺序/二进制保持。独立编译baseline不冒充整模型实际CUBIN。仅组件，无模型接入或正式成绩。

M512/4096×三bank×三个预定种子/尺度阶段=18数值case；layer0无残差，20/41有残差。输入含全零行与phase1残差抵消行。完整norm按原预声明FP64/两次BF16舍入界复核，候选/对照A8及scale各自对其实际BF16 norm全量CPU重算逐位相同。每个case每臂取8行×16列共128个原INT32结果，以完整K2048的INT64点积及显式FP32乘scale/BF16舍入验证，共4608个projection参考单元；不冒称所有输出单元独立重算。完整Graph返回norm/residual/q/scale/projection/INT32对象单独保留、毒化后重放，全部字节必须等于同臂eager，input/weight/RNG/输出guard均核对。已证实字节相同的Graph/eager原始文件可在本试验结果目录内部硬链接，保持独立路径和实际Graph输出hash，避免重复存储；禁止指向活跃输入/代码/模型文件。

完整链计时包含全部norm、A8、native mm及dequant；不能只测norm再宣传完整链收益。每bank/M使用phase0，eager及单图链长4/16，共6形状×3执行方式×2臂×5轮=180行，交替臂顺序、固定warm10/replay20、排首轮后中位数。主要门槛：M4096三种实际bank/残差情形的eager和chain16完整链均改善>3%，且各臂chain4/16差距<=10%；M512诊断不放宽门槛。所有数值/guard/Graph及独立CPU核验通过才有组件线索，不能替代新鲜完整模型质量与同期4K。

GPU probe尚未实现，不在此轮RMS+LM质量/短测/条件正式队列内；必须等待26165/27948/30742全部结束且GPU空闲才允许后续独立组件。原始预计约1.4GiB，运行前要求至少2.3GiB余量，当前数据不删除。截止15:58Z硬停，源码/配置在首次GPU前冻结。原始模型/数组/缓存/完整日志只天数容器，GitHub仅代码/报告/关键门槛。
