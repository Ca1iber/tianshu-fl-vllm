# 天数32g：在新KV16/warp1/B64S8基础上复核直接Decode映射

旧20261012实验在早期warp4基线上，B64算子快约1.3%，未做模型服务。本次基线已变为L0、guarded KV16/warp1与B64八分段，因此单独复核，不沿用旧收益。

仅改变单query Decode producer的CTA到sequence映射：读取cu_seqlens定位实际query，跳过0-query填充；grid从queries//BLOCK_Q+num_seqs缩到num_seqs。浮点计算、KV循环、reducer、tile/warp/分段与当前基线一致；原空CTA会提前退出，不声称省掉相同比例的KV读写。Prefill不走候选。基线和候选都保持无sink分母L0。

CPU映射2736组0/1 query模式通过；只证明逻辑query覆盖及无重复写，不证明Triton或性能。33个GPU场景待测：B1/2/8/16/24/32/40/48/56/64×4K/5K，10个短KV边界，以及3个strided/0-query padding案例。实际交错KV、随机页表、FP32/CPU参考、output/scratch尾部和9轮交替Graph事件计时。暂未验证改变Graph输入，也未完成真实模型路由或新Math。

串行控制器准备等待K/V平面组件成功退出、归档完成及GPU空闲后执行一次微测。没有模型配置采用，没有完整benchmark、16K服务或安装源码改写。若没有明确微测收益，不再启动模型对照。
