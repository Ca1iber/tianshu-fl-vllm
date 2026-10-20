# 新prefill Math102/105，正式4K运行中

预约到2026-10-22 00:00北京时间；各GPU任务2026-10-21T15:58:00Z硬停止。GPU唯一当前队列仍prefill_int8_validation，official根 /workspace/benchmark-results/20261021_prefill_int8_official4k_tianshu，Math根同前缀_math_tianshu。Math已102/105=97.142857%，105成功0错误、incomplete=false，104stop+1max_tokens（索引83仍正确，全部计分）；355190输出tokens，实际新prefill M256覆盖42层。质量独立归档到tianshu_prefill_int8_quality_results_20261021并已push。

正式4K N256×4轮排首轮：R1 5420.72 total/1084.14 output/TTFT5676.02排除；R2 5583.62/1116.72/3834.25，256/256，后两轮继续，不先下最终成绩。当前默认仍5482.79/Math101，最高吞吐备选5509.23/Math102；新正式必须独立复核CSV/stdout/407来源/12主controllers/62env、现场tune与同一新Math后再决定采用。新复核器在独立quality/official_results目录，各自SHA，不改已冻控制器。release干运行407/12/62通过，26份生成依赖原字节。

新增两个组件准备均已commit并完成CPU检查，尚未有GPU成绩：tianshu_int8_down_grouped_20261021（Down真实W，9M×12配置，基线真实mm保留旧M2048专用tile，并附测vendor linear）；tianshu_k8vi8_fixedk_pairv_20261021（4路径，恒1 K-scale消除及成对V8 load，三路V8保持8448B物理页/同writer）。CPU导入vLLM若GPU可见会建上下文，首轮期间断言发现后已隔离为CUDA_VISIBLE_DEVICES=-1且no_GPU_initialized通过；计分轮不做可见GPU的CPU准备。

run_after_prefill_official_components.sh等待已有official结果complete且退出0、新Math>95%、validation父runner/complete.txt，再唯一顺序Down GPU→不可见GPU的CPU复核→KV GPU→CPU复核。队列会保存失败且停止，不会与正式并行，不会自动采用或覆盖默认。其runner /workspace/benchmark-results/20261021_after_prefill_components_runner_tianshu，queue同名前缀.log/.pid；查看实际阶段/PID，别重复启动。两个组件全raw仅容器，关键gate/CSV/rounds/report推GitHub。

后续优先：每个正式轮完成报成绩，全部结束运行official_results/verify_official.py（CUDA不可见的CPU），保存原CSV/summary/comparison，若adopted则更新TIANSHU_BEST/发布manifest与tag并push；继续看Down/KV组件结果。没有本配置新16K，不安排全16K。用户两个gateup_silu dirty文件始终不碰。
