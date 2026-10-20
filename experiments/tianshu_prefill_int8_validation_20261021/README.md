# 同一407运行来源的新Math与正式4K（仅准备）

入口run_validation.sh只在当前4K小测配对完成、增幅>0.5%且大于两组内波动、计分轮没有现场调优时允许运行。先完整105题Math，成功无错误、逐题评分与索引复核、严格>95%，再原始benchmark脚本N256×4轮排除首轮。未排队或启动GPU，不做常规16K。

服务模型使用小测的原407来源/62环境/13参数/PYTHONPATH；验证器是额外CPU controller，分别冻结/快照，不把Math换成另一份运行profile。Math与正式4K必须有相同runtime sources、归一化环境和controller。纯decode NN、原生权重/内存/attention/Graph/FutureKV等按原质量审计保持；Math核验C8真实请求，正式核验C64/4K真实请求。两者均必须有42层大M INT8实际路径，Math不强求M4096或C64请求，明确其作用域；没有人为加GPU波次改变质量工作负载。

独立适配旧质量审计的BF16 prefill覆盖条件，替换为实际INT842层证据；保留fallback BF16实现、原warmup、packing、源SHA和其他检查。当前尚未过屏幕门，准备代码不等于启动或通过。
