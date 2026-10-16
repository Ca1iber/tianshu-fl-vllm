# 几何TopP候选：离线准备全保留103题的续跑缓存

当前预约按2026-10-16 18:00结束处理，本轮不连接或启动GPU。上一轮真实Math103/105在17:58按时停止，完整质量及正式成绩仍未取得。

只读检查容器实际安装的EvalScope1.11.1源码四份：`arguments.py`、`config.py`、`api/evaluator/cache.py`、`evaluator/evaluator.py`。`CacheManager.filter_prediction_cache`按题号恢复预测并从待生成数据中排除这些样本；`Evaluator._collect_work_items`把已有预测转成review-only工作项，剩余样本走predict+review。`--use-cache`配合`--rerun-review`用于复用全部预测并重新评分。该机制来自本容器源码，不外推其他版本。

原103题评分位于`.jsonl.rerun-UUID`事务临时文件，正常完成前未发布为正式review缓存。准备副本时，仅把103条预测和原task_config逐字节复制到新缓存，事务评分只放在provenance保存，不写入正式reviews或reports。下次实际续跑必须重新评分105题，不把临时文件冒充完成的评测报告。

`prepare_resume_cache.py`已对真实终态原件运行成功：58份源快照及当前仓库源、38项实际API进程环境、固定配置SHA和数据指纹一致；103条预测与评分题号/题目/消息内容对应、模型无API错误、单choice及停止原因正常。每条输出的reasoning与text原样保留。评分字符串只包含可见text，核验时读取text片段进行对照，不改写缓存内容。完整预测文件SHA `b3aef70457201b723b0d09cf69bfed3995fe0ac3ddc2aa32c03d45f04697c2ce`。

本机缓存：`service-results/20261016_tianshu/gemm_topp_geo_math_resume_seed`，明确`complete_quality=false`、`evaluation_started=false`、`installed_EvalScope_integration_verified=false`。103题预测逐字节保留，包括原有错误回答；缺index83/101。manifest记录全部103行SHA、原生成参数/命令、58源SHA、数据指纹及4份缓存实现的固定SHA。

真实数据正例和5项拒绝检查均通过：重复已完成预测、改变top_p、混入历史其他版本答案、改变EvalScope缓存实现、试图覆盖现有输出均被拒绝。全部检查在临时副本上进行；原终态快照的每一个文件字节均保持不变。具体结果见`CPU_AUDIT_32g.json`。

```bash
python3 experiments/tianshu_gemm_topp_geo_20261016/prepare_resume_cache.py \
  --snapshot-root ../service-results/20261016_tianshu/gemm_topp_geo_math_terminal/20261016_gemm_topp_geo_math_tianshu \
  --cache-sources ../service-results/20261016_tianshu/evalscope_cache_sources \
  --output /tmp/a-new-math-cache-directory
```

仍未验证的前提：实际安装EvalScope完整类型/数据集加载器对该副本的接受情况、CLI工作目录与缓存身份恢复流程，以及实际105题汇总。下次预约后先在容器中验证这些项和全部目标运行源码/参数，再以可丢弃的工作副本续跑，原103预测不得变化、删题或替换。只有最终0..104全105条唯一有效预测和重新评分、report请求与成功数105/零错误并且完整正确率大于95%，才能继续本候选正式4K N256×4；不因本离线准备修改正式最佳3992.97或声称新正确率。

CPU调用栈诊断仍仅准备未运行。用户授权的上版GitHub推送已完成，本轮没有再次push。
