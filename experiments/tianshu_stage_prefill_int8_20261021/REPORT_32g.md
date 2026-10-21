# 当前版本阶段测量：采集完成，canary检查失败，保留部分诊断

原实验维持 `failed / AssertionError()`，没有新的模型性能或质量通过结论。正式最好5586.28与Math102/105不改。

当前407份正式源码逐项相同；原13argv与62环境项保留，额外观察hook仅改变PYTHONPATH入口。417份准备源冻结，测量中发现的原生llama.py另入快照后为418份。128个4K/512负载请求全部完成，两批64请求分别43.352696s、39.166800s；这些有插桩的响应时长不换算正式tok/s。

40步均为FULL/padded64、64个实际请求，context从min4128/max4192至min4167/max4231。测量使用Graph外事件，预分配事件，仅边界两次同步。stage文件完整，源码快照与事件结构在独立CPU恢复中验证。

|区间|40帧中位数 ms|累计区间占比|
|---|---:|---:|
|total|36.397106|100%|
|forward|32.255121|88.559994%|
|logits|0.610931|1.684744%|
|sample|3.403963|9.375642%|

host execute wall/thread CPU中位数4.029756/4.020339ms，sample_tokens为1.977233/1.978936ms。异步调用重叠，CPU与GPU不能相加；stream区间包括host提交空隙，不是各kernel活跃时间。以上只作定位线索，不作为已经通过完整审计的profile。

## 实际失败与保留边界

前后canary均同128输入、同seed777、16个生成token、usage正确，但文本出现差异：前为`LEVEL errorest faut rememberedNOTH그통서보췄ո`，后为`LEVEL errorest faut rememberedNOTH그통서보쥭ո`。驱动在文本相等断言处拒绝，之后的after_metrics与完整runtime_audit均未执行。不能用原Math或源码相同把这次检查改成成功，也不能凭此断言观察器改变模型结果。

另发现原未运行的verify_profile.py对model_type预期为LlamaForCausalLM，实际observer记录的是vllm_fl.compilation.graph.GraphWrapper。核心Llama compute_logits的实际文件另已哈希快照；冻结源码不再修改，该元数据差异在CPU恢复中显式记录。

独立恢复目录 `../tianshu_stage_prefill_int8_failure_recovery_20261021` 仅重读已保存JSON/源码/响应/事件，输出PARTIAL_DIAGNOSTIC_32g.json与40帧；`original_experiment_passed=false`、`canary_equal=false`、`full_runtime_audit_completed=false`、`after_metrics_missing=true`。没有GPU重跑、重写原result/state或补造metrics。ready后新autotune记录为0，结束GPU空闲，runner237761终态Z。

CPU准备首次因407正式清单含两个同hash的ModelRunner路径而拒绝；v2精确检查两个路径集合及各自hash，全部通过后才运行GPU。初次CPU日志/快照均留容器。

可能原因待验证：原INT8 prefill按M选择路径，M128可走INT8而缓存命中后的M16等走BF16；同prompt重复可能改变计算形状。此为假设，正在单独准备无observer、原407源的cache_salt控制诊断，不能提前宣称因果已成立。也已查到早期直接decode CTA映射和INT8 GUP+SiLU模型接入的负收益历史，避免把已试方向当作新发现。

完整原件：`/workspace/benchmark-results/20261021_stage_prefill_int8_tianshu`。原result SHA256 `f6e36487d985be1f7fb582e8e2f0601bd4bf2bb1ce1ea4d44f07fa5982f635b2`，stage SHA256 `6c82fdd1982fc6402bf31e322fecd9343cf3cc20082a86cf33e8897833ebfc4c`。准备提交3936ceb已推送。GitHub仅保存代码、报告和小型诊断，完整回答/日志只在容器。

后续独立无observer对照已完成：`../tianshu_prefill_cache_canary_v2_20261021/REPORT_32g.md`。4冷/4热组内token一致，冷/热第13、14token不同，实际hits为0/112。原stage失败状态不变；新增证据说明不能把这次文本差异直接归因observer，后续canary需使用独立冷缓存。
