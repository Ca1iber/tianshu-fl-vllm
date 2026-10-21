# 融合路径高并发补充质量：103/105通过

同全部105题Level3、T1/top_p.95/max32768，batch64。103/105=98.095238%，105请求均成功、0错误，103正常stop与2达到32768。原代码冻结后未修改，没有重复GPU挑分，也不复用旧版102/105。

这项是官方batch8的补充，不替换官方Math8：后者101/105=96.190476%。两者使用相同450项数值源/13项原serve参数；补充只有FutureKV准入candidate→control，保持原生allocator，真实最多64活跃请求。KV40141blocks，实际42层M64融合重放和LM64 INT8均通过冻结审核。

C64_QUALITY_GATE_32g.json约83KB，保存105题完整执行摘要、runtime核验结论、450源映射及原始题目回答的SHA；原始回答、逐层重复记录保留在容器。root=/workspace/benchmark-results/20261021_nn_silu_fx_c64_quality_tianshu。

两项质量门均过，watcher249716已自动启动正式4K原命令N256/C64四轮排首轮，恢复canonical FutureKV candidate。正式controller255255/API255257，root=/workspace/benchmark-results/20261021_nn_silu_fx_official4k_tianshu。截至报告写入时尚在starting_server，没有新吞吐纪录。正式最好仍5586.28 total/1117.25 output、TTFT3780.63ms。

用户最新要求是先完成这一轮，之后逐环节嗅探M范围内重排/融合收益，再生成分角色范围白名单。准备源/计划在../tianshu_m_range_stages_20261021，GPU driver尚未完成；当前不并行启动新GPU任务。
