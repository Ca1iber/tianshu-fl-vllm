# 天数32g：int32 attention 完整验证结论

同一新profile，新Math102/105（97.142857%）、0请求错误，四轮原始4K与16K均全部成功。仅把native Decode内部stride/element offset算术缩为32，并在B4..48/BF16/NONE和offset<2^31下选择；浮点计算与64物理指针保留，旧Prefill/bucket/gather/Stage1设置不变。

|正式工作负载|旧profile tok/s|新profile tok/s|吞吐变化|旧TTFT ms|新TTFT ms|TTFT变化|
|---|---:|---:|---:|---:|---:|---:|
|4K/1K C64 N256|3222.52|3202.05|−0.635217%|4510.23|4501.53|−0.192895%|
|16K/1K C64 N128|2440.62|2600.48|+6.549975%|200982.18|189964.79|−5.481775%|

新profile4K后三轮波动5.121100%，16K后三轮波动0.103827%。16K有明确实测收益；4K这次小幅低于旧纪录，仍需同期A/B判断原因，不能仅凭两次均值断言算子负优化。保留旧4K纪录/launcher与新16K独立验证配置，记录按profile归属，避免合并不同配置的成绩。完整raw、源SHA、独立Math匹配、每轮数据、CSV及不可变archive SHA见official4k_32g/official16k_32g/math_32g子报告。

其它整数探索：radix histogram32/16/8和private index32微测有收益但服务无收益，真实大batch原生top-p已32；KV写入整数32多无收益，Prefill launch w4/s1约29→12.6us仅微测，未替换实际模型。独立SQL目标表反射实测冷首轮258.12→106.72s，暖态3253.60→3257.60（+0.123%，无明确稳态增益），未混入本次正式profile。

报告与每轮通知已按用户要求完成，原始benchmark未修改，沐曦仓库只读参考，未push远程GitHub。运行工具run_math.py/run_official.py/run_followups.py及全部源快照保留，预约截止前可继续独立优化实验。
