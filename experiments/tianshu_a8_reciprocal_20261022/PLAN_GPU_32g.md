# 快速A8实际GPU逐项核验

CPU枚举19796种BF16 peak、81084416点，简单倒数乘法7424个整数码不同；half-div与half-fma修正0不同，仍需真实GPU证明。

GPU覆盖所有>=eps的正有限BF16 peak，每个peak含指数差0..9的128个mantissa及两符号，排除abs(A)>peak，宽4096；另加全部25688个clamp区域有限BF16数。其余指数差>=10有127*(255/128)*2^-10<0.25，浮点余量远小于距0.5的距离，均应量化0。旧量化器与两修正完整保存q/scale逐项核验。NaN/Inf不在有限BF16等价合同内，原除法fallback保留。

性能：M8/K2048、M64/K2048、M64/K6144、M17/K259 strided、M512/K2048、M4096/K2048，均4warp/1stage。同一Graph三相位变更输入，q/scale位级相同、guards/输入/RNG保持。每图32次量化，7轮轮换，每轮10次replay再除以320，减少Python单kernel提交间隙；属于量化器微测，非模型吞吐。

独立CPU不导入GPU框架，从位模式核验覆盖，用FP32除法与FP64→FP32→RNE分别核对完整结果。现用NT/NN的_quantize和wrapper AST相同要单独检查。精确且有速度收益才准备框架接入、短4K及新Math。没有新增权重副本，原checkpoint不读写。

源冻结后不改，失败另目录。每任务30分钟；每天北京时间06:00–24:00，本日23:58截止。raw仅容器，GitHub代码/报告/关键结果。
