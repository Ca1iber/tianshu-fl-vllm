# 天数 BI-V150 32g：B16分段策略额外短边界通过

继22长短边界/66候选扫描后，独立补B16、warp2的14短/填充场景：KV1/7/13/15/16/17/31/32/33/63/64/65，以及非连续Q/KV和0-query填充33/65。S4对照，候选S2/S8/S16共42项全部通过选定CPU独立参考、GPU FP32、native固定容差、输出/S64 scratch尾部与Graph重放；40/42逐位native相同。实际gridZ和warp2/9轮Graph原始样本已核验。较短KV下S16可能慢，只作为边界验证，不声称所有长度均更快。

probe源95b0ea09c7824704c2f11ddda22cb3de238d9af24ed2989e8a8d966d4413c4de，router0ebd0726，与所有源snapshot逐字一致；L0生成4a1f8b及native bab2不变。全原始包2511458bytes/10分片，SHA256 9458204cda873732af30b64c6601a95d72d8f772cdf413835c3bfce2d5721cbe，分片/整包、42实际结果及源/实际grid/timing本地核验。原件service-results/20261014_tianshu/decode_segments_b16_short。

当前正式最好3640.16/728.03/TTFT4465.17/newMath103/tag tianshu-best-decode-kv32-w2-20261014保持。准备仅B1 S64→32、B16 S4→16，其他batch原策略和warp不变的4K小测；实际服务路由/ROI仍需另证，必要时新独立Math105严格>95，不复用现103。尚无新吞吐分数，未采集图内硬件profile，TODO；无新16K服务、安装源/权重/benchmark不变、沐曦只读。
