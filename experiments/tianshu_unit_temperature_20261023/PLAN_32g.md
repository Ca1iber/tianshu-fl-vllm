# 实际请求 T=1 的中性温度缩放

r4 GPU时间线观察每步FP32 rank1 true_div约191us。候选不按比赛硬编码温度，实际InputBatch每次batch更新生成metadata时从CPU temperature_cpu[:num_reqs]判断全体精确等于1且all_random，附在此次metadata；采样期间以ContextVar确认相同metadata与temperature tensor，仅FP32/CUDA跳过除以1，否则调用原函数。控制组也安装同样观察/判定，但仍执行原除法。不读GPU数值、不改变采样参数/RNG/TopP/权重/调度/命令，r4原944来源SHA不改。metadata变动重新判断，未知meta自动fallback。

前置GPU：实际r4采样栈，温度1/.7/1.4/混合，B1/8/64，global与部分/全部每请求generator；完整温度缩放结果逐位、最终token及global/每请求RNG状态一致，BF16 logits原输入不变。flag判断测试新metadata及未知meta回退，不能用缓存flag跨metadata。通过后才3轮N64/4K/1K同次服务筛选，剔除首轮，完整响应和原始时长保留。纯单位温度必须实际观察到C64跳过；control跳过为0。尚无新Math与正式N256资格，不替换r4。截止北京时间23:58。
