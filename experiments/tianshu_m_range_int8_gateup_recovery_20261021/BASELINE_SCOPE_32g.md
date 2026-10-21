# 组件基准范围：不能把独立激活的收益直接当模型收益

原result/CSV中的current_total对M64正确保留现有two_dot融合；其他M为NN GEMM + FlagGems独立SiLU组件链，模型真实动态prefill却由Inductor生成SiLU。current_model_INT8_enabled只表示相应M当前已用W8A8，不代表组件链就是模型实际全部kernel。

M4096单独NN GEMM约1.9528ms、独立FlagGems激活约1.0675ms，组合3.0612ms；融合2.2559ms。融合虽比该拆开组件链快35.70%，仍比单独GEMM慢，若Inductor实际激活更快，模型收益可能消失。M128对应.09244/.04050/.13263/.10188ms也有相同问题。单kernel微图还会受CPU提交间隙影响，不能简单把各项独立时间相加。

所以先用从当前prefill生成代码提取且AST不变的Inductor激活，轮换42实际权重并包含BF16 Down，测完整独立MLP；组件重新编译不声称原服务binary相同。之后仍需真实4K/新Math，当前未采用该范围融合，正式最好仍5618.01 total tok/s。

M64+0.15%最佳差异不构成新收益；GROUP_M在只有一个M tile时调度本就等价，不以重复测量噪声选版本。未启用INT8的稀有M表格只算INT8子系统探索，不据此扩大模型精度白名单。
