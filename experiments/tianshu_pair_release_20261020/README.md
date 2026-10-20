# 成对 K 读取版本的直接启动入口

正式4K已通过独立复核与采用：5235.52 total tok/s、1047.10 output tok/s、平均TTFT4161.08ms，4轮排除首轮，1024/1024请求成功；独立新Math103/105（98.095238%）。本配置没有完整16K新成绩，3885.91属于前版Future-KV配置。正式证据见../tianshu_pair_official_results_20261020/OFFICIAL_GATE_32g.json。

在原比赛镜像中运行：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_pair_release_20261020/restore_generated_sources.sh --check
bash experiments/tianshu_pair_release_20261020/serve_tianshu_minicpm.sh
```

若容器重建导致生成Python缺失，去掉--check先恢复。清单仅包含26份Python源码，现有文件如SHA不符就退出，不覆盖冲突。原模型、驱动和安装包仍由比赛镜像提供；代码包不含模型、编译缓存或完整回复。

入口以Bash/jq校验282份运行来源与8份控制文件，清除继承的实验开关，导出54项已测量配置，再直接exec原始vllm serve的13项argv。TIANSHU_PAIR_K_PROFILE_RECORD_DIR可指定记录目录，TIANSHU_PAIR_K_PROFILE_DRY_RUN=1仅校验，不启动GPU。

DRY_RUN_GATE_32g.json证明入口的源码/控制文件SHA、实际导出环境、污染开关清理及带空格记录路径通过CPU演练；它不代表额外一次模型或GPU性能评测。RELEASE_MANIFEST_32g.json把此入口与已实测原命令的同一配置关联。

该正式版本没有包含后续W8A8、短分段或uint32读取实验。版本标签tianshu-original-command-4k-k8v16-pairk-20261020记录正式成绩检查点，后续文档补充保持同一启动配置。
