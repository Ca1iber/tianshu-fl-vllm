# 天数32g：standalone TopP精度修复及大分块实验

针对前一完整原件发现的普通/peaked误差和ties质量.99518，独立生成native64c666源副本：仅standalone TopP用exact概率重复统计helper，概率二分停止宽度1e-12/最多32次；常数输入起始范围退到0；最终在概率域判断pivot和重复值，避免log反算logit。rowstart预置4标量alias防止未定义SSA。原TopK分支AST保持，新Router对所有TOPK_ENABLED都明确回退原native/8192/4096/w16/s1，不因候选tile修改combined语义。源变换可逐字逆还原原native。

上述是组合修复假设，不预先宣称根因或所有数值已修好。CPU实际Router12组合、TopK回退和可逆源/原TopK AST核对通过（假JIT非GPU）。GPU10输入包括既有7场景、B64 BF16往返normal/peaked和B8全部相等；6计划含native baseline/identity及修复8k16w/16k16w/16k32w/4k16w。修复候选在原输入及flip输入必须独立严格CPU mass/minimality/order1e-6通过才计时，不允许用exactbaseline跳过已有误差。保存逐行记录与实际sourceactive/tile/warp/编译产物，7x10事件。

尚无新GPU/服务/Math/正式结果；当前4091.29配置不改。若修复可靠且更快再做新服务对照/独立确认/新Math与正式，不把精度修复自动当模型提速。
