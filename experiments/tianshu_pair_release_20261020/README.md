# 成对 K 读取版本的直接启动入口

当前为候选发布准备，正式采用以 ../tianshu_pair_official_results_20261020/OFFICIAL_GATE_32g.json 的 adopted 为准。独立新 Math103/105 已通过；完整4K仍在运行，本配置未跑完整16K。

在原比赛镜像运行 restore_generated_sources.sh --check，然后运行 serve_tianshu_minicpm.sh。若容器重建导致生成 Python 缺失，去掉 --check 先恢复。清单仅包含26份 Python 源码；现有文件如 SHA 不符就退出，不覆盖冲突。模型、驱动和安装包仍由比赛环境提供。

入口以 Bash/jq 校验282份运行来源与8份控制文件，清除继承的实验开关，导出54项已测量配置，再直接 exec 原始 vllm serve 的13项 argv。TIANSHU_PAIR_K_PROFILE_RECORD_DIR 可指定记录目录；TIANSHU_PAIR_K_PROFILE_DRY_RUN=1 仅校验，不启动GPU。CPU演练证明配置一致，不代表另外完成一次模型或性能评测。

正式复核已完成：4K5235.52 total tok/s、1047.10 output tok/s、TTFT4161.08ms，独立采用证明 adopted=true。对应新Math103/105，完整4K1024/1024请求成功。本配置没有完整16K新成绩；3885.91是前版Future-KV配置的已测值。
