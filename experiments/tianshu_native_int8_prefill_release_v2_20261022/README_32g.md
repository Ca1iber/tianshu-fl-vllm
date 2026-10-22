# 天数原生INT8预填充：4K推荐版

正式4K：**5777.22 total / 1155.45 output tok/s，平均TTFT 3459.44 ms**。相对旧5695.95吞吐+1.43%，TTFT降低5.58%。新C8与真实C64均101/105（96.1905%）。

原官方4K/1K C64/N256四轮排首轮，1024请求全部成功，热轮差幅0.6353%，零抢占，就绪后Triton autotune日志0。16K本版未复测。

在原天数比赛容器使用：

```bash
cd /workspace/vllm-plugin-FL
bash experiments/tianshu_native_int8_prefill_release_v2_20261022/restore_frozen_cache.sh --check
TIANSHU_NATIVE_INT8_PROFILE_DRY_RUN=1 bash experiments/tianshu_native_int8_prefill_release_v2_20261022/serve_tianshu_minicpm.sh
bash experiments/tianshu_native_int8_prefill_release_v2_20261022/serve_tianshu_minicpm.sh
```

入口直接用Bash执行原vllm命令，实际工作目录固定/workspace。启动前校验晋升门槛、877模型源、20控制器、384 cache文件所属冻结源、原13argv和63项实际环境。只M512/4096换原生INT8，M8/M64、LM和其他M保持旧路径，原模型/架构不变，临时192MiB workspace由原生profile计费，无新增静态权重。

若/root中的原source cache丢失但/workspace的同hash冗余副本仍在，可不带--check补齐缺失文件，再检查。已有文件hash不符不会覆盖；若两份都缺失，需要在相同比赛镜像按已有源码预热/校验流程重建，未把编译二进制大文件上传GitHub，也未宣称仅git clone就能恢复全部cache。

C8原GPU覆盖审计失败、正式原GPU收尾缺workload字段失败均保留原phase和hash。C8按真实BF16范围独立复核；新C64真实INT8全42层通过；正式workload由原命令和四轮CSV恢复，仅补CPU审核metadata，4K零抢占与全部原检查均保留，OFFICIAL_GATE明确original_experiment_passed=false。

DRAFT_PLAN为准备阶段记录；最终资格以OFFICIAL_GATE、DRY_RUN_GATE和RELEASE_MANIFEST为准。详细数值与范围见../tianshu_native_int8_prefill_official_results_v2_20261022/REPORT_32g.md。原5695.95发布入口仍保留。
