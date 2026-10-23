# prefill完整QKV链：三组真实权重输入已独立核验，GPU组件尚未实现

只读原MiniCPM5-2B checkpoint，从layers0/20/41各自q/k/v拼接实际2560×2048 QKV权重。纯NumPy从BF16原始位解码、计算FP32逐行scale及nearest-even对称INT8，三组原BF16、INT8 NT数据和scale的SHA都与r3实际已合格bank逐字节一致。原checkpoint SHA仍14fb8e7f0a18d53d1f239773758bf581cee7e456a4523a54622c3a245b64402c。三份input norm原BF16权重共12KiB仅存容器CPU目录，无完整权重或checkpoint副本，不导入Torch/Triton/vLLM/FlagGems。

下一组件预定比较完整的IR-native norm→A8→原native INT8 QKV→dequant链，候选只合并norm与A8，其他weight pack/GEMM/dequant保持；M512/4096、三组实际bank/残差情形和三个数值阶段共18用例。GPU probe和独立CPU verifier尚未实现，也未运行GPU；norm输入准备和既有CUBIN记录不是新M4096数值或速度结果。

计划在源码/GPU控制器冻结后验证完整norm FP64界、A8/scales全量重算、4608个完整K INT64投影参考单元、真正Graph输出毒化/重放、guard与输入/权重/RNG，随后完整链eager与chain4/16计时，不以norm单独加速推断投影或模型吞吐。本准备不加入当前RMS+LM质量/短测/条件正式队列，不影响其冻结模型或已记录计时，没有采用或发布。

代码、元信息、计划和本报告放天数GitHub；三份norm输入仅/workspace/benchmark-results/20261023_prefill_RMS_A8_QKV_CPU_tianshu。GPU运行前至少2.3GiB余量，必须等26165/27948/30742链全部结束且GPU空闲。
