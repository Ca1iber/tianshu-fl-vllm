# BI-V150 32G：KV16 Decode stage1 独立质量通过

在已验证 Native GQA L0 Prefill、Hybrid routing、adaptive Decode、预算2048基础上，仅将匹配3D unquant/Q16-KV2/head128/B<=64的原生Decode启动从KV16/w4/stage3改为KV16/w4/stage1。原kernel body、grid、分块和数学公式保留。模型与已安装源码未修改。

本次独立运行官方 Math-500 Level 3 的105题：102正确，97.142857%，105请求成功且无请求错误，105 unique reviews均success；严格满足accuracy>95%。temperature=1.0、top_p=0.95、max_tokens=32768、batch8、原始数据与chat endpoint保持官方命令。三个错误索引18、87、23；第23题达到32768输出上限，其余104题stop。没有复用原版本102/105的结果。

真实路由记录14类Decode batch，均KV16/w4/stage1、shared12288；68条Native Prefill记录均M256/KV128/w16/s1、无sink初值0。准确率审计记录两hook、生成器、原生kernel、Hybrid/adaptive/worker SHA与原始评测命令；后续正式4K必须匹配这些源码与预算。

小规模4K/1K、C64/N128三轮排首轮3186.657122 total tok/s，相对同规格控制2742.829558提升16.181376%，三轮均无新增调优。该协议与正式N256四轮不同，不能据此升级正式成绩。当前独立准确率与原正式最好2712.55对应质量均为102/105，尚无本候选完整正式吞吐。

原始预测、review、report、日志、路由及源码快照全量本地备份：2818522 bytes，11分块和整包SHA256 d0192ea12f6fd6ecc468dabaeb50be7875f348578a42713b8c273f6057eb39c7核验；预测/review/report各自SHA再次核验。保留本次真实预测，不重建响应。

正式4K原脚本4轮/C64/N256/排首轮已启动；当前正式最好仍为2712.55 total、542.51 output tok/s、TTFT4720.38ms，较公开2028.01 total基线提升33.75%。性能剖析TODO，未采集该候选端到端profile；未跑优化版完整16K。
