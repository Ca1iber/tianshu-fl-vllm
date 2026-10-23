# 天数量化进展（2026-10-23 北京时间17:53）

当前采用的正式最佳仍为r3：4K 5950.75 total /1190.15 output tok/s、TTFT3174.58ms。标签tianshu-original-command-4k-native-int8-attention-prefill-20261022-r3，commit9044c25。没有新16K成绩。

单Down64 W4 v2完成新鲜完整质量：C8 102/105（97.14%）、C64 103/105（98.10%），均严格>95%，两轮各1个max_tokens全部保留官方review及105分母。完整210题/原始SHA/916模型来源/85控制器/实际profile独立复核通过。真实42个W4 M64捕获与主模型M64执行链接，旧Down M64 BF16捕获0；其他M Down、原QKV/O64 BF16、LM64原W8A8均保留。原checkpoint未变。

其同场N64短4K对照5877.47→5910.54 total tok/s（+0.5628%），通过固定筛选。随后单次原始N256×4/exclude1正式4K热轮5985.30/5992.38/5984.34，均值5987.34 total /1197.47 output tok/s、TTFT3240.21ms、吞吐波动0.1343%。相对历史r3吞吐+0.6149%，TTFT+2.0674%，固定TTFT上限1%未过，故不采用、不构建合格发布、不改r3标签。1024请求全部成功、零运行期autotune/抢占、全部源与运行审计/质量/原始CSV/checkpoint独立CPU重算通过。详见tianshu_dots_Down64_official_results_v2_20261023。

限制：正式第2/3轮同时进行了另一候选的CPU文件/源码准备，没有CPU隔离；TTFT波动不能全部归因于W4，历史r3也不是同场完整AB。后续计时期间停掉后台CPU准备。保留本结果，不原样重跑追分。

| 已验证组件 | 实际当前对照→候选 | 组件收益，非整机成绩 |
|---|---|---|
| 42层Down64 AWQ W4，graph | 2.320458→2.188805ms | +6.01%，省0.131654ms/解码步 |
| 42层Down64 dotsaddz W4，graph | 2.320545→1.940144ms | +19.61%，省0.380401ms/步 |
| LM64 AWQ W4，相对实际已用W8A8核心eager | 0.611353→0.535017ms | +14.27%，省0.076336ms |
| LM64 dotsaddz W4，同对照 | 0.611353→0.519834ms | +17.61%，省0.091519ms |

Down完整2359296 FP64格、LM两W4完整50135040 FP64格通过，含真实权重、全字节/版本/guard/RNG、eager/graph poison核对。LM对照是已采用b9b692自写quantize+mm核心（预分配偏向对照），没有错用未采用的原生INT8或BF16。这是group128对称RTN与原生W4后端，不是AWQ激活校准，投影误差约13%仍须模型质量验证。

当前下一候选在单Down W4上新增LM64 W4：同934模型来源/95实际GPU控制器，新增142049280B副本，真实load与native scratch探针完成。新鲜C8原版本正常complete为102/105=97.14%，C64为101/105=96.19%，后者原审计因继承Down的LM64 INT8期望遗漏而audit_failed；原失败保留、未GPU重跑，通过单独更严格W4实证CPU校正。完整双质量210题、全部来源/快照/raw SHA/checkpoint与实际运行审计独立重算通过，model_quality_qualified=true。详见tianshu_dots_LM64_quality_results_v2_20261023，尚无本组合吞吐成绩。

CPU研究保留group32/64/128/256及非对称minmax误差表，未GPU或模型测试的新group/参数不混入当前候选。此前prefill RMS/QKV融合虽C8/C64均101/105，但短4K仅+0.0507%、波动1.0088%，未采用。Down W4 v1因新capture0/旧BF16capture42在startup拒绝，零题评测；原失败证据与来源保留。

容器中仅已结束实验快照按相同文件名和完整内容去重，7965文件中7173重复文件合并硬链接释放515.73MiB，CPU准备快照另释放84.43MiB，所有路径/原字节SHA均复核通过。没有硬链接实时源文件、删除原始答案或搬数据到用户本地。代码/报告/关键门禁和CSV推天数GitHub，模型/大数组/编译缓存/预测全文均只在容器。MuXi仓库只读参考、未修改。

C64的105请求全部正常stop，无cap，错误索引87/42/18/23。独立CPU校正见tianshu_dots_LM64_quality_recovery_20261023；原result仍audit_failed，新独立目录CPU_recovery_complete且绑定原失败SHA与CPU代码SHA。完整质量v2已两轮全部通过；旧results v1和short v1没有用于后续GPU，不能绕过校正语义。

后续使用tianshu_dots_LM64_short4k_v2_20261023：原r3精确903来源对934来源，N64×3/exclude1、固定>0.5%/spread<1%/TTFT<=control1.01。必须完整新C8与校正后的C64均>95%才开始；计时期间不做后台CPU准备。条件正式控制器tianshu_dots_LM64_official4k_v2_20261023及独立formal_results、发布入口草稿已准备已CPU预检冻结，尚未GPU执行或采用；先完成CPU准备冻结，再开始任何计时，不在short或正式轮次中同时编代码/打包。

18:10更新：Down64+LM64 W4完成同场短4K。热轮r3 5888.47→候选5911.96 total tok/s（+0.3988%），candidate spread0.5491%，TTFT8576.20→8603.67ms（+0.3203%）。全部384请求/完整运行与独立CPU审计通过，CPU准备均在计时前完成，但增益未达到预设0.5%。因此不跑已准备的正式控制器、不构建发布、不打tag，正式仍r3；不原样重跑追分。下一步独立研究W4解包/反量化与矩阵乘组件实现，native组件收益不等于服务收益。

## 18:32，Triton W4解包组件负结果

32个配置全部编译并通过独立数值审核；42层真实Down64的Graph最快13.525959ms，原生W4约1.973ms，BF16约2.320ms。没有新模型/正式成绩，不采用。详情见`../tianshu_triton_W4_Down64_20261023/REPORT_32g.md`。同时完成容器可重建FX图缓存完整归档校验后清理，恢复约4.1GiB磁盘，源码和384项固定缓存校验通过，详情见`../tianshu_fxgraph_storage_20261023/REPORT_32g.md`。
