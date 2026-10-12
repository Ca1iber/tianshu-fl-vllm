# BI-V150 32G：流式输出间隔1与4

保持Native L0 Prefill、Hybrid、adaptive Decode KV16/w4/stage1、预算2048、buckets/gather关闭。只把服务stream_interval从默认1改4，减少发送块次数。原始benchmark未修改；这是本地4K/1K、C64/N64三轮小测，排首轮，不能当N256正式成绩。

控制均值3240.773553、候选3258.100402 total tok/s，变化+0.534652%。TTFT13473.856373→13487.731433ms，基本持平。两组各3×64请求，384usage均4096输入/1024输出，配对各轮提示seed列表一致，两组全部三轮调优0/0/0。没有明确>5%提升，不采用、不新跑Math/正式。

完整原始日志、响应、路由与实际执行脚本本地备份3230450 bytes、13分块及整包SHA256 6f48690aa58ef0ba8d3498fbe3f841e337ccdc4c1027cfdd5cd81cc31b467ff0全部核验。实际运行script另保存快照，后续75点实验对公共trial的参数扩展不追溯改变此结果。未采集新profile。
