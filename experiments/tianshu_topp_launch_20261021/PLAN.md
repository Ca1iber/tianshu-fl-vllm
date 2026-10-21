# 原生 Qrita top-p launch 配置组件

假设：BI-V150原生top-p采用单program/行、多遍scan，默认warp4不一定最佳。保留完整原始JIT函数、BLOCK8192/TRUNC4096、TopK关闭和全部TopP/duplicate/NaN逻辑，仅探索warp2/4/8/16。不声称消除现有full sort，因为C>=8当前已经是Triton Qrita。无RNG、模型或新准确率成绩。

实际原生Python wrapper作baseline，tap记录其真实compiled binary/warp/tiles。显式warp4必须binary SHA与baseline相同。B8/B64×6分布normal1/normal4/peaked/uniform/masked/nonfinite=12场景，三次改变完整输入与p（第三phase含0/.1/.5/.9/.9999/1），5模式180个完整输出检查。原生异常/全等logits语义保留，不删除场景或放宽native等价要求；候选与原生不等者完整记录为不等，不冒充数值成功。全部retained值bitwise原输入、padding/arena guard、RNG与原输入不变必须通过。

每phase保存完整BF16输入、p、全部输出mask bitpack与完整FP32输出SHA；独立CPU可重建每个完整输出并比SHA、复核所有mask和原生差异。CPU理想nucleus对照仅作额外诊断，原生已有异常/舍入行为不能据此被隐藏或把新配置错误改判成功。原始完整mask数值与性能都需CPU归档核查之后才决定服务。

计时7轮轮换×20hot Graph replay，每次Graph内部先从不变源完整恢复未过滤FP32 logits，所有arm都包含相同copy成本；防止反复过滤后的空输入产生假速度。计时不含RNG/完整softmax/模型，不能直接换算服务收益。超过native mask全等的正收益才优先考虑；先模型小4K再决定Math/正式，禁止只凭best-of候选宣称提升。仍需实际服务logits验证。

仅唯一GPU任务，最多30分钟并受北京时间23:58截止。CPU gate和所有py/sh/PLAN/primary源在GPU启动后不可修改，新代码修复必须新dir；CPU元数据归档另开目录。原始数据只留容器，GitHub只有代码/报告/关键JSON/CSV。
