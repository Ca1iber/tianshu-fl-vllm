# 14:54 当前阶段诊断与缓存canary因果核对结束，继续优化

预约10-22 00:00止，GPU23:58硬停；现在仍有约9小时。正式推荐不变521c154/tag tianshu-original-command-4k-prefill-int8-20261021，407源/62env/原13argv，4K5586.28总/1117.25out tok/s、TTFT3780.63ms、Math102/105。没有新正式成绩，不做例行16K。

本轮新stage目录tianshu_stage_prefill_int8_20261021准备3936ceb：沿用旧Graph外事件核心但链入当前完整hook，原407源子集/62env不改；417源准备，运行发现llama.py后418源。128负载响应全部成功，40帧FULL/C64完整，但前后seed777/128prompt/16out canary文本不同，因此原phase failed，after_metrics和runtime_audit未运行。冻结不改，不重跑。单独failure_recovery目录CPU重算40帧和128usage，gate明确original_experiment_passed=false、full_runtime_audit_completed=false。阶段中位数total36.397106/forward32.255121/logits.610931/sample3.403963ms，累计比例88.56%/1.68%/9.38%，仅部分诊断，非单kernel时间。实际model_type是GraphWrapper，原未运行verifier误预期Llama，不改冻结源，恢复中显式记录。cbff0be失败报告已推送。

无observer原407源缓存诊断V1在第一个HTTP200后full usage dict断言失败，原响应未保存，零完成对照，056f2d0保留失败。V2 tianshu_prefill_cache_canary_v2_20261021先保存响应再逐项核3个计数+16tokenIDs；CPU原生UsageInfo验证额外prompt_tokens_details字段，实际null。代码5fec90c已推送。

V2于14:50:48 complete，独立CPU校验及冻结hash通过，runner239191 Z，API239204/core239219消失，GPU空闲。8请求同128prompt/seed777/temp1/top_p.95/out16；unsalted首次/重复和A/B/C独立salt的cold/cached共4冷4热。实测每次queries128、hits0或112。4冷序列相同、4热相同，组间第13/14token cold137,248 / cached120,277不同。原42层M128 prefill_A8W8各4次、M16 delegate_qualified各4次，符合缓存改变prefill计算路径，但没有隔离各算子贡献，不把全部差异只归INT8。后续新服务A/B前后canary用不同cache_salt并核hits=0，避免冷热错比；正式benchmark/Math命令不改，旧stage仍failed，不补造metrics或声称全audit。

CACHE_CANARY_GATE/ROUTE_EVIDENCE/REPORT已准备本次推送，raw仅/workspace/benchmark-results/20261021_prefill_cache_canary_v2_tianshu。

后续候选应优先前向。已重新查历史避免重复：K8V16 direct_map_20261019网格96→64在4K基本持平、9920694已闭环，暂不重做。INT8 GateUp+SiLU已有旧NT组件13.67%收益，但MLP新opaque边界V2/V3服务比原参考慢4.67%，不采用；原父INT8 MLP route不能直接照搬。当前后来采用NN物理W，M64配置bm64/bn256/bk128/w4/s1，M8 bm32/bn256/bk128/w4/s1；NN+SiLU配对列/双dot融合尚需查是否已有，而新集成必须保留prefill原编译路径，避免把原activation在opaque fallback中变成eager。若探索，先CPU检查真实编译接口及融合可行性，再新NN组件完整数值和计时，有实际收益再服务。native vllm/config/compilation.py compile_sizes文档明确对应静态[i,i]范围，现模型日志是动态range(1,4096)，不能假定普通Python M8/64分支在现一次编译中安全生效；考虑真实compile_sizes/FX局部pass，尚未写新融合代码或运行GPU。不要把未实施的设计当完成。

旧结果：pair_lateV/addr32四路112GPU逐位相同但4K全部慢，505872c已推；T16七路全部慢a81ad03；QKV prefill服务+.293182%未达>.5%，6c4775e，不新Math；integerPV全部慢44298f3；TopP16K质量99/105失败不复测。其他旧负收益沿前STATE，不随意重复。

源码/报告/关键结果全容器+GitHub；本地不留副本，用户dirty gateup_silu两文件不碰，FlagGems4cdc3b7保持，沐曦只读，无子agent。SSH master/tmp/tianshu-20261021-control在16:27前刷新，先建立新连接成功再关旧连接；当前连通。下一turn直接读此状态与实际进程，继续前向/框架优化，goal保持active。
